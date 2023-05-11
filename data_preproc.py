import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm
import argparse
import logging
from scipy.optimize import minimize_scalar
from numba import jit
import time

# ORGANIZATION OF THE DATA (FOR EACH TRADING DAY)
# Message file's columns are as follows:
# Time, Event type, Order ID, Size, Price, Direction
# where Eventy type is:
# 1: Submission of a new limit order
# 2: Cancellation (partial deletion) of a limit order
# 3: Deletion (total) of a limit order
# 4: Execution of a visible limit order
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
    orderbook = pd.concat([pd.read_csv(f'{paths_lob[i]}') for i in range(N_days)])
    message = pd.concat([pd.read_csv(f'{paths_msg[i]}') for i in range(N_days)])
    print(orderbook)
    print(message)
    logging.info(f'Number of trading days considered: {N_days}\nTotal events: {orderbook.shape[0]}')
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

    # Select the bid and ask prices and volumes
    bid_prices = orderbook[[f'Bid price {x}' for x in range(1, int(n/4)+1)]]
    bid_volumes = orderbook[[f'Bid size {x}' for x in range(1, int(n/4)+1)]]
    ask_prices = orderbook[[f'Ask price {x}' for x in range(1, int(n/4)+1)]]
    ask_volumes = orderbook[[f'Ask size {x}' for x in range(1, int(n/4)+1)]]

    return orderbook, message, m, M, bid_prices, bid_volumes, ask_prices, ask_volumes

def executions_finder(message):
    # Select the market orders
    visible_lo = message[message['Event type'] == 4]
    hidden_lo = message[message['Event type'] == 5]
    # Select the buy and sell market orders
    # buy_side_exec = executions[executions['Direction'] == 1]
    # sell_side_exec = executions[executions['Direction'] == -1]
    executions = [visible_lo, hidden_lo]

    return executions#, buy_side_exec, sell_side_exec

# Define a function named 'dq' that takes as argument executions from executions_finder and finds:
# 1. All the rows of executions that share the same time
# 2. For each time, compute the max e min price
# 3. Compute (max-min)/tick_size

def dq_dist(executions, tick_size):

    dq = []
    
    for exec in executions:
        t = exec['Time'].value_counts().index
        c = exec['Time'].value_counts().values
        d = {'Time': t, 'Count': c}
        df = pd.DataFrame(data=d)
        for i in tqdm(range(df.shape[0]), desc='Computing dq'):

            # if executions[executions['Time'] == df['Time'][i]]['Direction'].unique().shape[0] != 1:
            #     raise ValueError('There are both buy and sell market orders at the same time')
        
            min = exec[exec['Time'] == df['Time'][i]]['Price'].min()
            max = exec[exec['Time'] == df['Time'][i]]['Price'].max()
            # print(exec[exec['Time'] == df['Time'][i]][['Price', 'Direction']])
            # time.sleep(1)
            if exec[exec['Time'] == df['Time'][i]]['Direction'].iloc[0] == 1:
                tick_shift = int((min - max) / tick_size)
            else:
                tick_shift = int((max - min) / tick_size)

            dq.append(tick_shift)
    
    return dq

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
        X[np.where(mask_ask)] = list(d_ask.keys())
        tick_positions = np.nonzero(X)[0]
        tick_labels = p_line[tick_positions].astype(int)

        plt.figure(tight_layout=True, figsize=(15,5))
        plt.title(f'Limit Order Book {event}')
        plt.bar(np.arange(p_line.shape[0]), volumes, width=2.5, color=colors)
        plt.vlines(mid_price_loc, 0, volumes.max(), color='black', linestyle='--')
        plt.xlabel('Price')
        plt.ylabel('Volume')
        plt.xticks(tick_positions, tick_labels, rotation=90)
        plt.xlim([min_bid_loc - 10, max_ask_loc + 10])
        plt.savefig(f'lob_snapshot_{event}.png')
        lob_snapshots.append(imageio.imread(f'lob_snapshot_{event}.png'))
        plt.clf()
        os.system(f'rm lob_snapshot_{event}.png')
        plt.close()

    imageio.mimsave('lob_snapshots.gif', lob_snapshots, fps=10)

def mid_price(bid_prices, ask_prices):
    return 0.5 * (bid_prices['Bid price 1'] + ask_prices['Ask price 1'])

def sq_dist_objective(f, midprice, target_var=2):
    midprice = midprice / 10000
    var = midprice[::int(f)].diff().var()
    print(var)
    sq_dist = (var - target_var)**2
    return sq_dist


def avg_imbalance(bid_volumes, ask_volumes, weight, f):
    lev = weight.shape[0]
    events = bid_volumes.shape[0]
    imb = []
    for i in tqdm(range(0, events, f)):
        num = np.dot(np.array(bid_volumes.iloc[i:i+f, :lev]), weight).sum()
        det = np.dot((np.array(bid_volumes.iloc[i:i+f, :lev]) + np.array(ask_volumes.iloc[i:i+f, :lev])), weight).sum()
        imb.append(num/det)
    return np.array(imb)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script is used to plot the limit order book and the mid price for a given trading day''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-TSLA", "--TSLA", action='store_true')
    parser.add_argument("-MSFT", "--MSFT", action='store_true')
    parser.add_argument("-N", "--N_days", type=int, help='Number of trading days', default=100)
    parser.add_argument("-ip", "--imbalance_plot", action='store_true', help='Plot the imbalance distribution')
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    # plt.style.use('ggplot')

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

    dq = np.zeros(1)
    orderbook, message, m, M, bid_prices, bid_volumes, ask_prices, ask_volumes = data_preproc(paths_lob, paths_msg, args.N_days)
    executions = executions_finder(message)
    midprice = mid_price(bid_prices, ask_prices)
    res = minimize_scalar(sq_dist_objective, args=(midprice,), bounds=(1, 1000), method='bounded')
    print(res)
    exit()
    DQ = np.array(dq_dist(executions, 100))
    dq = np.hstack((dq, DQ))
    np.save('dq.npy', dq)
    value, count = np.unique(dq, return_counts=True)
    x = np.arange(value.min(), value.max()+1)
    mask = np.in1d(x, value)
    y = np.zeros_like(x)
    y[mask] = count/count.sum()
    plt.figure(tight_layout=True, figsize=(5,5))
    plt.bar(x, y, color='black')
    plt.show()

    
    if args.imbalance_plot:
        plt.figure(tight_layout=True, figsize=(15,5))
        weight = np.array([1,1,0,0])
        f = 10
        imb = avg_imbalance(bid_volumes, ask_volumes, weight, f)
        plt.hist(imb, bins=100, histtype='bar', density=True, alpha=0.5, label=f'{weight}')
        weight = np.array([0,0,1,1])
        imb = avg_imbalance(bid_volumes, ask_volumes, weight, f)
        plt.hist(imb, bins=100, histtype='bar', density=True, alpha=0.5, label=f'{weight}')
        plt.legend()
        plt.show()


# Personal notes
#WHICH PRICE IS THE BEST TO USE? MID PRICE OR THE PRICE OF THE FIRST LEVEL?
# Fix the concatanation of the dataframes in data_preproc
# Compute the best frequency of sampling with a target price variance of 2.


# Note: in the case of monitoring spoofing this is better to be skipped, since
# I would remove just information.
# Intuition behind the estimations I have to do:
# In order to monitor, I have to estimate what is the distribution of the imbalance, both
# the legitimate and the spoofing one. In order to do so, I need the optmial weights for
# each level in the LOB. To obtain them, I have to minimize a function of the probabillity
# of having a certain price deviation dp given a certain imabalance (with certain weights) times
# the probability of having that imbalance.
# A cosa mi serve dq?