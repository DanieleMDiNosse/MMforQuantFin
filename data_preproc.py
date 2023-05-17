import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm
import argparse
import logging
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from numba import jit
import time
import warnings
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm

# ORGANIZATION OF THE DATA (FOR EACH TRADING DAY)
# Message file's columns are as follows:
# Time, Event type, Order ID, Size, Price, Direction
# where Eventy type is:
# 1: Submission of a new limit order
# 2: Cancellation (partial deletion) of a limit order
# 3: Deletion (total) of a limit order
# 4: Execution of a visible limit orderread_csvread_csv
# 5: Execution of a hidden limit order
# 6: Cross
# 7: Trading halt indicator
# Instead, Direction is:
# -1: Sell limit order
# 1: Buy limit order
# Note: Direction = 1 means that a buy order has initiated a trade, i.e. an agent has sold 
# causing the price to decrease. Viceversa, Direction = -1 means that a sell order has
# initiated a trade, i.e. an agent has bought causing the price to increase.
# Orderbook file's columns are as follows:
# Ask price 1, Ask size 1, Bid price 1, Bid size 1, Ask price 2, Ask size 2, Bid price 2, Bid size 2, ...
# For TSLA the tick size in 2015 was 0.01 USD (100 in the data)

def data_preproc(paths_lob, paths_msg, N_days):

    if (N_days > len(paths_lob)) or (N_days > len(paths_msg)):
        warnings.warn(f'\nNumber of days considered is greater than the number of days available. Number of days considered: {N_days}. Number of days available: {len(paths_lob)}. N_days is set to {len(paths_lob)}.')
        N_days = len(paths_lob)

    # create the list of dates
    dates = [paths_lob[i].split('/')[-1].split('_')[1] for i in range(N_days)]
    datetimes = []
    for day in dates:
        year, month, day = int(day.split('-')[0]), int(day.split('-')[1].split('0')[1]), int(day.split('-')[2])
        datetimes.append(datetime(year, month, day))

    orderbook = [pd.read_pickle(f'{paths_lob[i]}') for i in range(N_days)]
    message = [pd.read_pickle(f'{paths_msg[i]}') for i in range(N_days)]

    for i in range(N_days):
        message[i][message[i].columns[0]] = message[i][message[i].columns[0]].apply(lambda x: datetimes[i] + timedelta(seconds=x))
        orderbook[i].columns = [f'dummy_column_{i}' for i in range(orderbook[i].shape[1])]
        message[i].columns = [f'dummy_column_{i}' for i in range(message[i].shape[1])]
    message = pd.concat(message, ignore_index=True)
    orderbook = pd.concat(orderbook, ignore_index=True)
    logging.info(f'--------------------\nNumber of trading days considered: {N_days}\nTotal events: {orderbook.shape[0]}\n------------------------------')
    # Drop the last column of the message dataframe if there are > 6 columns
    if message.shape[1] > 6: message = message.drop(columns=message.columns[-1])

    n = orderbook.shape[1]
    ask_price_columns = [f'Ask price {i}' for i,j in zip(range(1, int(n/2)+1), range(0,n, 4))]
    ask_size_columns = [f'Ask size {i}' for i,j in zip(range(1, int(n/2)+1), range(1,n, 4))]
    bid_price_columns = [f'Bid price {i}' for i,j in zip(range(1, int(n/2)+1), range(2,n, 4))]
    bid_size_columns = [f'Bid size {i}' for i,j in zip(range(1, int(n/2)+1), range(3,n, 4))]
    ask_columns = [[ask_price_columns[i], ask_size_columns[i]] for i in range(len(ask_size_columns))]
    bid_columns = [[bid_price_columns[i], bid_size_columns[i]] for i in range(len(ask_size_columns))]
    columns = np.array([[ask_columns[i], bid_columns[i]] for i in range(len(ask_size_columns))]).flatten()
    orderbook.columns = columns

    message.columns = ['Time', 'Event type', 'Order ID', 'Size', 'Price', 'Direction']
    # Select the minimum and maximum values of the prices
    m, M = orderbook[ask_price_columns[-1]].min(), orderbook[bid_price_columns[-1]].max()

    return orderbook, message, m, M

def executions_finder(message):
    # Select the market orders
    visible_lo = message[message['Event type'] == 4]
    # hidden_lo = message[message['Event type'] == 5]
    executions = visible_lo#, hidden_lo]

    return executions

# Define a function named 'dq' that takes as argument executions from executions_finder and finds:
# 1. All the rows of executions that share the same time
# 2. For each time, compute the max e min price
# 3. Compute (max-min)/tick_size

def dq_dist(executions, tick_size, depth):

    dq = []
    t = executions['Time'].value_counts().index
    c = executions['Time'].value_counts().values
    d = {'Time': t, 'Count': c}
    df = pd.DataFrame(data=d)
    for i in tqdm(range(df.shape[0]), desc='Computing dq'):
    
        min = executions[executions['Time'] == df['Time'][i]]['Price'].min()
        max = executions[executions['Time'] == df['Time'][i]]['Price'].max()

        if executions[executions['Time'] == df['Time'][i]]['Direction'].iloc[0] == 1:
            tick_shift = int((min - max) / tick_size)
        else:
            tick_shift = int((max - min) / tick_size)

        if np.abs(tick_shift) > depth:
            pass
        else:
            dq.append(tick_shift)

    return np.array(dq)

def lob_reconstruction(N, tick, m, M, bid_prices, bid_volumes, ask_prices, ask_volumes):
    n_columns = bid_prices.shape[1]
    lob_snapshots = []
    for event in tqdm(range(N)):
        p_line = np.arange(m, M+tick, tick)
        volumes = np.zeros_like(p_line)
        d_ask = {ask_prices.iloc[event][i]: ask_volumes.iloc[event][i] for i in range(int(n_columns))}
        d_bid = {bid_prices.iloc[event][i]: bid_volumes.iloc[event][i] for i in range(int(n_columns))}
        mid_price = bid_prices['Bid price 1'][event] + 0.5*(ask_prices['Ask price 1'][event] - bid_prices['Bid price 1'][event])

        # Create two boolean arrays to select the prices in the p_line array that are in the bid and ask prices
        mask_bid, mask_ask = np.in1d(p_line, list(d_bid.keys())), np.in1d(p_line, list(d_ask.keys()))

        # Assign to the volumes array the volumes corresponding to the the bid and ask prices
        volumes[np.where(mask_bid)] = list(d_bid.values())
        volumes[np.where(mask_ask)] = list(d_ask.values())

        # Insert the mid price in the p_line array and the corresponding volume in the volumes array
        max_bid_loc = np.array(np.where(mask_bid)).max()
        min_bid_loc = np.array(np.where(mask_bid)).min()
        min_ask_loc = np.array(np.where(mask_ask)).min()
        max_ask_loc = np.array(np.where(mask_ask)).max()
        mid_price_loc = max_bid_loc + int(0.5 * (min_ask_loc - max_bid_loc))
        p_line = np.insert(p_line, mid_price_loc, mid_price)
        volumes = np.insert(volumes, np.array(np.where(mask_ask)).min(), 0)

        bid_color = ['g' for i in range(p_line[:mid_price_loc].shape[0])]
        ask_color = ['r' for i in range(p_line[mid_price_loc:].shape[0])]
        colors = np.hstack((bid_color, ask_color))

        X = np.zeros_like(p_line)
        X[np.where(mask_bid)] = list(d_bid.keys())
        # print(np.where(mask_ask)[0] + 1)
        X[np.where(mask_ask)[0] + 1] = list(d_ask.keys())
        tick_positions = np.nonzero(X)[0]
        tick_labels = p_line[tick_positions].astype(int)

        plt.figure(tight_layout=True, figsize=(15,5))
        plt.title(f'Limit Order Book {event}')
        plt.bar(np.arange(p_line.shape[0]), volumes, width=1, color=colors)
        plt.vlines(mid_price_loc, 0, volumes.max(), color='black', linestyle='--')
        plt.xlabel('Price')
        plt.ylabel('Volume')
        plt.xticks(tick_positions, tick_labels, rotation=90)
        plt.xlim([min_bid_loc - 10, max_ask_loc + 10])
        # plt.show()
        plt.savefig(f'lob_snapshot_{event}.png')
        lob_snapshots.append(imageio.imread(f'lob_snapshot_{event}.png'))
        plt.clf()
        os.system(f'rm lob_snapshot_{event}.png')
        plt.close()

    imageio.mimsave('lob_snapshots.gif', lob_snapshots, fps=24)


def sq_dist_objective(f, orderbook, tick, target_var=2):
    ask_price = orderbook['Ask price 1'] / tick
    ask_price_diff = ask_price[::f].diff()
    print(ask_price_diff.head(50))
    sq_dist = (ask_price_diff.var() - target_var)**2
    return sq_dist


def avg_imbalance(num_events, bid_volumes, ask_volumes, weight, f):
    lev = weight.shape[0]
    imb = []
    for i in tqdm(range(0, num_events, f), desc='Computing average imbalance'):
        num = np.dot(np.array(bid_volumes.iloc[i:i+f, :lev]), weight).sum()
        det = np.dot((np.array(bid_volumes.iloc[i:i+f, :lev]) + \
                      np.array(ask_volumes.iloc[i:i+f, :lev])), weight).sum()
        imb.append(num/det)
    return np.array(imb)

def joint_imbalance(message, imbalance):
    message = message.iloc[:imbalance.shape[0]]
    executions = executions_finder(message)
    t = executions['Time'].value_counts().index
    c = executions['Time'].value_counts().values
    d = {'Time': t, 'Count': c}
    df = pd.DataFrame(data=d)
    i_m, i_p = [], []
    for i in tqdm(range(df.shape[0]), desc='Computing joint imbalance'):
    
        min = np.array(executions[executions['Time'] == df['Time'][i]].index).min()
        max = np.array(executions[executions['Time'] == df['Time'][i]].index).max()
        i_m.append(imbalance[min])
        i_p.append(imbalance[max])

    plt.figure(tight_layout=True, figsize=(6,6))
    plt.scatter(i_m, i_p, s=15, c='black', alpha=0.8)
    plt.xlabel(r'$i_{-}$')
    plt.ylabel(r'$i_{+}$')
    plt.show()

    return np.array(i_m), np.array(i_p)
    
    
def direction_of_hlo(message):
    hlo = message[message['Event type'] == 5]
    hlo = hlo['Direction'].unique()
    print(hlo)
    return hlo

def dp_dist(parameters, M, num_events, bid_volumes, ask_volumes, f):
    dp = parameters[:M + 1]
    w = parameters[M + 1:]
    imb = avg_imbalance(num_events, bid_volumes, ask_volumes, w, f)
    shape, loc, scale = stats.skewnorm.fit(imb)
    obj_fun = 0
    for m in range(M):
        obj_fun += - 1 / M * np.log((imb[-1] * dp[m] + (1 - imb[-1]) * dp[-m]) \
                                     * stats.skewnorm.pdf(imb[-1], shape, loc, scale))
    return obj_fun

def callback(x, M, N, bid_volumes, ask_volumes, f, iteration):
    print("Iteration :\n", iteration)
    print(f"Current solution for dp:\n{x[:M + 1]} -> sum = {x[:M + 1].sum()}")
    print(f"Current solution for w:\n{x[M + 1:]} -> sum = {x[M + 1:].sum()}",)
    print("Objective value:\n", dp_dist(x, M, N, bid_volumes, ask_volumes, f))
    print("-------------------------------")
    obj_fun.append(dp_dist(x, M, N, bid_volumes, ask_volumes, f))

def create_callback(M, N, bid_volumes, ask_volumes, f):
    iteration = 0
    def callback_closure(x):
        nonlocal iteration
        iteration += 1
        callback(x, M, N, bid_volumes, ask_volumes, f, iteration)
    return callback_closure

def constraint_fun1(x):
    return x[:x.shape[0] - 4].sum() - 1

def constraint_fun2(x):
    return x[x.shape[0] - 4:].sum() - 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script is used to plot the limit order book and the mid price for a given trading day''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-TSLA", "--TSLA", action='store_true')
    parser.add_argument("-MSFT", "--MSFT", action='store_true')
    parser.add_argument("-imb", "--imbalance_plot", action='store_true', help='Plot the imbalance distribution')
    parser.add_argument("-pp", "--pre_proc", action='store_true', help='Perform data preprocessing')
    parser.add_argument("-dq", "--dq_dist", action='store_true', help='Compute the distribution of dq')
    parser.add_argument("-dp", "--dp_dist", action='store_true', help='Compute the distribution of dp')
    parser.add_argument("-lob", "--lob_reconstruction", action='store_true', help='Reconstruct the limit order book')
    parser.add_argument("-f", "--freq", action='store_true', help='Compute the optimal frequency of sampling')
    parser.add_argument("-j", "--joint_imbalance", action='store_true', help='Compute the joint imbalance distribution')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    np.random.seed(666)
    # plt.style.use('ggplot')

    tick = 100 # Set the tick size. For TSLA (MSFT) the tick size in 2015(2018) was 0.01 USD (100 in the data)
    info = [['TSLA', '2015-01-01'], ['MSFT', '2018-04-01']]
    if args.TSLA:
        info = info[0]
    elif args.MSFT:
        info = info[1]

    if args.pre_proc:
        logging.info('Performing data preprocessing')
        N_days = int(input('Number of days to consider: '))
        paths_lob = []
        paths_msg = []
        if args.TSLA:
            folder_path = 'data/TSLA_2015-01-01_2015-01-31_10'
            filenames = os.listdir(folder_path)
            for file in filenames:
                if 'orderbook' in file:
                    paths_lob.append(f'/home/danielemdn/Documents/MMforQuantFin/{folder_path}/{file}')
                elif 'message' in file:
                    paths_msg.append(f'/home/danielemdn/Documents/MMforQuantFin/{folder_path}/{file}')
        elif args.MSFT:
            folder_path = 'data/MSFT_2018-04-01_2018-04-30_5'
            filenames = os.listdir(folder_path)
            for file in filenames:
                if 'orderbook' in file:
                    paths_lob.append(f'/home/danielemdn/Documents/MMforQuantFin/{folder_path}/{file}')
                elif 'message' in file:
                    paths_msg.append(f'/home/danielemdn/Documents/MMforQuantFin/{folder_path}/{file}')
        
        paths_lob.sort()
        paths_msg.sort()
        orderbook, message, m, M = data_preproc(paths_lob, paths_msg, N_days)
        orderbook.to_pickle(f'{info[0]}_orderbook_{info[1]}.pkl')
        message.to_pickle(f'{info[0]}_message_{info[1]}.pkl')
    


    orderbook = pd.read_pickle(f'{info[0]}_orderbook_{info[1]}.pkl')
    message = pd.read_pickle(f'{info[0]}_message_{info[1]}.pkl')
    period = [message['Time'].iloc[0].strftime('%Y-%m-%d'),  message['Time'].iloc[-1].strftime('%Y-%m-%d')]
    f = 1
    orderbook, message = orderbook.iloc[::f].reset_index(drop=True), message.iloc[::f].reset_index(drop=True)
    ret = (orderbook['Ask price 1'] / tick).diff()
    depth = np.quantile(ret[1:], 0.99)
    n = orderbook.shape[1]
    bid_prices = orderbook[[f'Bid price {x}' for x in range(1, int(n/4)+1)]]
    bid_volumes = orderbook[[f'Bid size {x}' for x in range(1, int(n/4)+1)]]
    ask_prices = orderbook[[f'Ask price {x}' for x in range(1, int(n/4)+1)]]
    ask_volumes = orderbook[[f'Ask size {x}' for x in range(1, int(n/4)+1)]]
    m, M = bid_prices.min().min(), ask_prices.max().max()
    logging.info('\nData loaded: {}.\nDataframe shape: {}.\nDepth: {}.\nPeriod: {} - {}'.format(info, orderbook.shape, depth, period[0], period[1]))


    if args.lob_reconstruction:
        N = int(input('Number of events to consider: '))
        lob_reconstruction(N, tick, m, M, bid_prices, bid_volumes, ask_prices, ask_volumes)

    if args.dq_dist:
        executions = executions_finder(message)
        dq = dq_dist(executions, tick, depth)
        np.save('dq.npy', dq)
        value, count = np.unique(dq, return_counts=True)
        print(value, count)
        x = np.arange(value.min(), value.max()+1)
        mask = np.in1d(x, value)
        y = np.zeros_like(x)
        y[mask] = count
        y = y / count.sum()
        plt.figure(tight_layout=True, figsize=(5,5))
        plt.bar(x, y, color='black')
        plt.show()
    
    if args.dp_dist:
        N = int(input('Number of events to consider: '))
        M = int(depth)
        initial_params = np.random.uniform(0, 1, M+5)
        print("Initial parameters:\n", initial_params)
        obj_fun = []
        constraint1 = {'type': 'eq', 'fun': constraint_fun1}
        constraint2 = {'type': 'eq', 'fun': constraint_fun2}
        constraints = [constraint1, constraint2]
        bounds = [(0,1) for i in range(M+5)]
        res = minimize(dp_dist, initial_params, args=(M, N, bid_volumes, ask_volumes, 1),  \
                       constraints=constraints, method='SLSQP', bounds=bounds, \
                        callback=create_callback(M, N, bid_volumes, ask_volumes, 1))
        print(res)
        print("Initial parameters:\n", initial_params)
        np.save(res.x[:M+1], 'dp.npy')
        np.save(res.x[M+1:], 'ws.npy')
        plt.figure(tight_layout=True, figsize=(5,5))
        plt.plot(obj_fun)
        plt.title('Objective function')
        plt.figure(tight_layout=True, figsize=(5,5))
        plt.bar(np.arange(M+1), res.x[:M+1], color='black')
        plt.title(r'$dp^+$')
        plt.show()
    
    if args.imbalance_plot:
        plt.figure(tight_layout=True, figsize=(15,5))
        weight = np.array([0.6,0.5,0.2,0.1])
        f = 1
        N = int(input('Number of events to consider: '))
        imb = avg_imbalance(N, bid_volumes, ask_volumes, weight, f)
        shape, loc, scale = stats.skewnorm.fit(imb)
        end = time.time()
        x = np.linspace(np.min(imb), np.max(imb), 100)
        plt.plot(x, stats.skewnorm.pdf(x, shape, loc, scale), 'r-', label=f'Skew Normal {shape, loc, scale}')
        plt.hist(imb, bins=100, histtype='bar', density=True, alpha=0.5, label=f'{weight}')

        weight = np.array([0.1,0.2,0.5,0.6])
        imb = avg_imbalance(N, bid_volumes, ask_volumes, weight, f)
        shape, loc, scale = stats.skewnorm.fit(imb)
        x = np.linspace(np.min(imb), np.max(imb), 100)
        plt.plot(x, stats.skewnorm.pdf(x, shape, loc, scale), 'g-', label=f'Skew Normal {shape, loc, scale}')
        plt.hist(imb, bins=100, histtype='bar', density=True, alpha=0.5, label=f'{weight}')
        plt.legend()
        plt.show()
    
    if args.joint_imbalance:
        N = int(input('Number of events to consider: '))
        weight = np.array([0.6,0.5,0.2,0.1])
        imb = avg_imbalance(N, bid_volumes, ask_volumes, weight, f=1)
        np.save('imbalance.npy', imb)
        i_m, i_p = joint_imbalance(message, imb)
        np.save('i_m.npy', i_m)
        np.save('i_p.npy', i_p)

        i_m, i_p = np.load('i_m.npy'), np.load('i_p.npy')

        # plot the autocorrelation function of i_m and i_p and show them in the same figure with two plots
        fig, ax = plt.subplots(2, 1, figsize=(15,5))
        fig.suptitle('Autocorrelation function of $i_{-}$ and $i_{+}$')
        sm.graphics.tsa.plot_acf(i_m, ax=ax[0], lags=100)
        sm.graphics.tsa.plot_acf(i_p, ax=ax[0], lags=100)
        plt.show()

    if args.freq:
        l = []
        for f in range(1, 100):
            l.append(sq_dist_objective(f, orderbook, tick))
            exit()
        l = np.array(l)
        plt.plot(l, label='Objective function')
        plt.vlines(np.where(l == l.min())[0][0], -0.2, l.max(), linestyles='dashed', label=f'Minimum at f = {np.where(l == l.min())[0][0]}')
        plt.legend()
        plt.show()
    
    


# Personal notes
#WHICH PRICE IS THE BEST TO USE? MID PRICE OR THE PRICE OF THE FIRST LEVEL?
# Fix the concatanation of the dataframes in data_preproc1 -> Done
# Compute the best frequency of sampling with a target price variance of 2.


# Note:
# - dp e dq sono indipendenti? Non ne capisco il motivo.
# - A cosa serve stimare la frequenza ottimale? Ai fini del rilevamento di spoofing su
# singoli asset meglio tenere tutte le informazioni che si hanno a disposizione, no?
# In fin dei conti non mi interessa fare il confronto fra i diversi asset.
# - E' corretto il modo che ho usato per valutare dp+?