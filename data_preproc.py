'''
Message file's columns are as follows:
Time, Event type, Order ID, Size, Price, Direction
where Eventy type is:
1: Submission of a new limit order
2: Cancellation (partial deletion) of a limit order
3: Deletion (total) of a limit order
4: Execution of a visible limit order
5: Execution of a hidden limit order
6: Cross
7: Trading halt indicator
Instead, Direction is:
-1: Sell limit order -> a sell order has initiated a trade, i.e. an agent has bought
                        causing the price to increase
1: Buy limit order -> a buy order has initiated a trade, i.e. an agent has sold 
                        causing the price to decrease

Orderbook file's columns are as follows:
Ask price 1, Ask size 1, Bid price 1, Bid size 1, Ask price 2, Ask size 2, Bid price 2, Bid size 2, ...

For TSLA the tick size in 2015 was 0.01 USD (100 in the data)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import seaborn as sns
import cv2
import os
from tqdm import tqdm
import argparse
import logging
from scipy.optimize import minimize
import multiprocessing as mp
import numba as nb
import time
import warnings
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm

def data_preproc(paths_lob, paths_msg, N_days):
    '''This function takes as input the paths of the orderbook and message files for a period of N_days and returns
    the orderbook and message dataframes stacked and with the appropriate columns names, the minimum and maximum 
    values of the prices and the number of trading days considered (via logging).
    
    Parameters
    ----------
    paths_lob : list
        List of paths of the orderbook files.
    paths_msg : list
        List of paths of the message files.
    N_days : int
        Number of trading days to consider.
    
    Returns
    -------
    orderbook : pandas dataframe
        Dataframe containing the orderbook data.
    message : pandas dataframe
        Dataframe containing the message data.
    m : float
        Minimum value of the prices.
    M : float
        Maximum value of the prices.'''

    if (N_days > len(paths_lob)) or (N_days > len(paths_msg)):
        warnings.warn(f'\nNumber of days considered is greater than the number of days available. \
                       Number of days considered: {N_days}. Number of days available: {len(paths_lob)}.\
                       N_days is set to {len(paths_lob)}.')
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
    logging.info(f'--------------------\nNumber of trading days considered: \
                  {N_days}\nTotal events: {orderbook.shape[0]}\n------------------------------')
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
    '''This function takes as input the message dataframe and returns the executions dataframe. The executions
    dataframe contains all the rows of the message dataframe that correspond to a market order.
    
    Parameters
    ----------
    message : pandas dataframe
        Dataframe containing the message data.
    
    Returns
    -------
    executions : pandas dataframe
        Dataframe containing the executions data.'''

    # Select the market orders
    visible_lo = message[message['Event type'] == 4]
    # hidden_lo = message[message['Event type'] == 5]
    executions = visible_lo#, hidden_lo

    return executions

def dq_dist(executions, tick_size, depth):
    '''This function takes as input the executions dataframe, the tick size and the depth and returns the
    distribution of dq. The distribution of dq is a list of integers that represent the number of ticks
    that the price has moved after a market order. LOBSTER message files is organized in such a way that
    all the rows that share the same timestamp correspond the same market order. Therefore, the function
    computes the minimum and maximum price for each set of timestamps associated to an execution and then 
    computes the number of ticks that the price has moved. If the number of ticks is greater than the depth, 
    that value is discarded.

    Parameters
    ----------
    executions : pandas dataframe
        Dataframe containing the executions data.
    tick_size : float
        Tick size.
    depth : int
        Depth of the LOB.
    
    Returns
    -------
    dq : numpy array
        Array containing the distribution of dq.'''

    dq = []
    timestamps = executions['Time'].value_counts().index    # Select the timestamps
    counts = executions['Time'].value_counts().values       # Select the number of executions for each timestamp
    data = {'Time': timestamps, 'Count': counts}            # Create a dictionary with the timestamps as keys and the counts as values
    df = pd.DataFrame(data=data)                            # Create a dataframe with the number of executions for each timestamp
    for i in tqdm(range(df.shape[0]), desc='Computing dq'):
    
        min = executions[executions['Time'] == df['Time'][i]]['Price'].min() # Compute the minimum price for each timestamp
        max = executions[executions['Time'] == df['Time'][i]]['Price'].max() # Compute the maximum price for each timestamp

        if executions[executions['Time'] == df['Time'][i]]['Direction'].iloc[0] == 1: # If the direction is 1, the price has decreased
            tick_shift = int((min - max) / tick_size)
        else:                                                                         # If the direction is -1, the price has increased
            tick_shift = int((max - min) / tick_size)

        if np.abs(tick_shift) > depth:                                                # If the number of ticks is greater than the depth, discard that value
            pass
        else:
            dq.append(tick_shift)

    return np.array(dq)

# Nuova funzione dq_dist
def dq_dist2(executions, orderbook, depth):
    dq = []
    timestamps = executions['Time'].value_counts().index    # Select the timestamps
    counts = executions['Time'].value_counts().values       # Select the number of executions for each timestamp
    data = {'Time': timestamps, 'Count': counts}            # Create a dictionary with the timestamps as keys and the counts as values
    df = pd.DataFrame(data=data)                            # Create a dataframe with the number of executions for each timestamp

    for i in tqdm(range(df.shape[0]), desc='Computing dq'):
        executions_slice = executions[executions['Time'] == df['Time'][i]]       # Select all the executions for the specific timestamp. 
                                                                                 # Remember that executions is the message file filtered by event type = 4
        total_volume = executions_slice['Size'].sum()                            # Compute the total volume for of the market order

        if executions_slice.index[0] - 1 < 0:
            start = 0
        else:
            start = executions_slice.index[0] - 1

        orderbook_slice = orderbook.iloc[start: executions_slice.index[-1] + 1]  # Select the orderbook slice corresponding to the market order

        vol, tick_shift, j = 0, 0, 1

        if executions[executions['Time'] == df['Time'][i]]['Direction'].iloc[0] == 1: # If the direction is 1, the price has decreased and I have to consider the bid side
            while vol <= total_volume and j <= int(orderbook.shape[1]/4):
                vol += orderbook_slice[f'Bid size {j}'].iloc[0]
                j += 1
            if vol == total_volume: 
                tick_shift = -(j - 1)
            else:
                tick_shift = -(j - 2)
        else:                                                                         # If the direction is -1, the price has increased and I have to consider the ask side
            while vol <= total_volume and j <= int(orderbook.shape[1]/4):
                vol += orderbook_slice[f'Ask size {j}'].iloc[0]
                j += 1
            if vol == total_volume: 
                tick_shift = (j - 1)
            else:
                tick_shift = (j - 2)

        if np.abs(tick_shift) > depth:                                                # If the number of ticks is greater than the depth, discard that value
            pass
        else:
            dq.append(tick_shift)
        
    return np.array(dq)

def lob_reconstruction(N, tick, m, M, bid_prices, bid_volumes, ask_prices, ask_volumes):
    '''This function takes as input the number of events, the tick size, the minimum and maximum values of the prices,
    the bid and ask prices and volumes and returns the limit order book snapshots for each event.

    Parameters
    ----------
    N : int
        Number of events.
    tick : float
        Tick size.
    m : float
        Minimum value of the prices.
    M : float
        Maximum value of the prices.
    bid_prices : pandas dataframe
        Dataframe containing the bid prices.
    bid_volumes : pandas dataframe
        Dataframe containing the bid volumes.
    ask_prices : pandas dataframe
        Dataframe containing the ask prices.
    ask_volumes : pandas dataframe
        Dataframe containing the ask volumes.
    
    Returns
    -------
    lob_snapshots : list
        List of the limit order book snapshots for each event.'''

    n_columns = bid_prices.shape[1]
    lob_snapshots = []
    os.system(f'rm lob_snapshots/*')
    for event in tqdm(range(N), desc='Computing LOB snapshots'):
        # Create the price and volume arrays
        p_line = np.arange(m, M+tick, tick)
        volumes = np.zeros_like(p_line)

        # Create two dictionaries to store the bid and ask prices keys and volumes as values
        d_ask = {ask_prices[event][i]: ask_volumes[event][i] for i in range(int(n_columns))}
        d_bid = {bid_prices[event][i]: bid_volumes[event][i] for i in range(int(n_columns))}
        mid_price = bid_prices[event][0] + 0.5*(ask_prices[event][0] - bid_prices[event][0])

        # Create two boolean arrays to select the prices in the p_line array that are also in the bid and ask prices
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

        # Create the colors array to color the bars of the plot
        bid_color = ['g' for i in range(p_line[:mid_price_loc].shape[0])]
        ask_color = ['r' for i in range(p_line[mid_price_loc:].shape[0])]
        colors = np.hstack((bid_color, ask_color))

        # Create the tick positions and labels
        X = np.zeros_like(p_line)
        X[np.where(mask_bid)] = list(d_bid.keys())
        X[np.where(mask_ask)[0] + 1] = list(d_ask.keys())
        tick_positions = np.nonzero(X)[0]
        tick_labels = p_line[tick_positions].astype(int)

        # Plot the limit order book snapshot
        plt.figure(tight_layout=True, figsize=(15,5))
        plt.title(f'Limit Order Book {event}')
        plt.bar(np.arange(p_line.shape[0]), volumes, width=1, color=colors)
        plt.vlines(mid_price_loc, 0, volumes.max(), color='black', linestyle='--')
        plt.xlabel('Price')
        plt.ylabel('Volume')
        plt.xticks(tick_positions, tick_labels, rotation=90)
        plt.xlim([min_bid_loc - 10, max_ask_loc + 10])
        plt.savefig(f'lob_snapshots/lob_snapshot_{event}.jpg')
        plt.close()

# # compute the average volume for each price level of the LOB
# def lob_average_volume(N, tick, m, M, bid_prices, bid_volumes, ask_prices, ask_volumes):

def lob_video(image_folder, info):
    '''This function takes as input the path of the folder containing the limit order book snapshots and returns
    the limit order book video.

    Parameters
    ----------
    image_folder : str
        Path of the folder containing the limit order book snapshots.
    
    Returns
    -------
    None.'''

    frame_width = 1500
    frame_height = 500
    fps = 24.0
    output_filename = f"LOBvideo_{info[0]}_{info[1]}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    image_files = sorted(os.listdir(image_folder))  # Sort files in ascending order

    img = []
    for image_file in tqdm(image_files, desc='Creating LOB video'):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (frame_width, frame_height))
        video_writer.write(image)

    cv2.destroyAllWindows()
    video_writer.release()

# @nb.njit
def sq_dist_objective(f, tick_ask_price, target_var=2):
    # ask_price_fmean = np.array([tick_ask_price[t-f:t].mean() for t in range(f, tick_ask_price.shape[0])])
    # ask_price_diff_var = np.diff(ask_price_fmean).var()
    # sq_dist = (ask_price_diff_var - target_var)**2
    ask_price_fsampled = tick_ask_price[::f]
    ask_price_diff_var = np.diff(ask_price_fsampled).var()
    # sq_dist = (ask_price_diff_var - target_var)**2
    return ask_price_diff_var

@nb.njit
def dot_product(a, b):
    '''This function takes as input two numpy arrays and returns their dot product.
    
    Parameters
    ----------
    a : numpy array
        First array.
    b : numpy array
        Second array.
    
    Returns
    -------
    result : float
        Dot product of the two arrays.'''

    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# @nb.njit
def avg_imbalance(num_events, bid_volumes, ask_volumes, weight, f, bootstrap=False):
    '''This function takes as input the number of events, the bid and ask volumes, the weight and the frequency
    and returns the average imbalance.

    Parameters
    ----------
    num_events : int
        Number of events.
    bid_volumes : numpy array
        Array containing the bid volumes.
    ask_volumes : numpy array
        Array containing the ask volumes.
    weight : numpy array
        Array containing the weight.
    f : int
        Frequency.
    
    Returns
    -------
    imb : numpy array
        Array containing the average imbalance.'''

    lev = weight.shape[0]
    imb = []
    for i in tqdm(range(0, num_events), desc='Computing average imbalance'):
        num = dot_product(bid_volumes[i:i+f, :lev][0], weight)
        det = dot_product((bid_volumes[i:i+f, :lev][0] + ask_volumes[i:i+f, :lev][0]), weight)
        imb.append(num/det)
    imb = np.array(imb)
    shape, loc, scale = stats.skewnorm.fit(imb)
    
    if bootstrap == True:
        n_iterations = 100  # Number of bootstrap iterations
        n_samples = len(imb)  # Number of samples in your data

        # Initialize arrays to store parameter values from each iteration
        shape_samples = np.zeros(n_iterations)
        loc_samples = np.zeros(n_iterations)
        scale_samples = np.zeros(n_iterations)

        # Perform bootstrapping and refit the skew normal distribution in each iteration
        for i in tqdm(range(n_iterations), desc='Computing standard errors'):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = imb[bootstrap_indices]
            shape_samples[i], loc_samples[i], scale_samples[i] = stats.skewnorm.fit(bootstrap_sample)

        # Calculate the standard errors of the parameters
        shape_error = np.std(shape_samples)
        loc_error = np.std(loc_samples)
        scale_error = np.std(scale_samples)

        np.save(f'imbalance_{info[0]}_{info[1]}_errors', np.array([shape_error, loc_error, scale_error]))
    
    np.save(f'imbalance_{info[0]}_{info[1]}', imb)
    np.save(f'imbalance_{info[0]}_{info[1]}_params', np.array([shape, loc, scale]))
    shape_error = 0
    loc_error = 0
    scale_error = 0

    return imb, np.array([shape, loc, scale]), np.array([shape_error, loc_error, scale_error])

def joint_imbalance(message, imbalance):
    '''This function takes as input the message dataframe and the imbalance and returns the imbalance
    just before and after a market order. The empirical joint imbalance is then the scatter plot of the two arrays.

    "This distribution represents the overall joint distribution of the imbalance right before and after each 
    market orders fitted to the overall data. We assume that these represent the stable behavior of the market
    without spoofing, and therefore representative of legitimate market orders."
    Parameters
    ----------
    message : pandas dataframe
        Dataframe containing the message data.
    imbalance : numpy array
        Array containing the imbalance.
    
    Returns
    -------
    i_m : numpy array
        Array containing the imbalance just before a market order.
    i_p : numpy array
        Array containing the imbalance just after a market order.
    '''
    message = message.iloc[:imbalance.shape[0]]
    executions = executions_finder(message)
    t = executions['Time'].value_counts().index
    c = executions['Time'].value_counts().values
    d = {'Time': t, 'Count': c}
    # Create a dataframe with the number of executions for each timestamp
    df = pd.DataFrame(data=d)
    
    # Create two arrays containing the imbalance just before and after a market order.
    # For istance, i_m is evaluated considering the last imbalance value just before the
    # beginning of the market order. The index of this last imbalance value is found by
    # selecting the minimum index of the executions dataframe for each timestamp and taking
    # the previous index value. A similar procedure is applied to i_p.
    i_m = [imbalance[np.array(executions[executions['Time'] == time].index).min() - 1] for time in tqdm(df['Time'][:-1], desc='Computing i_m')]
    i_p = [imbalance[np.array(executions[executions['Time'] == time].index).max() + 1] for time in tqdm(df['Time'][:-1], desc='Computing i_p')]

    return np.array(i_m), np.array(i_p)

def i_spoofing(i_m, dq, dp_p, weight, k, depth, bid_volumes, ask_volumes, message):
    '''This function takes as input the imbalance just before a market order, the distribution of dq, the distribution
    of dp, the weight, the level k where the spoofing is performed, the depth, the bid and ask volumes and the message
    dataframe and returns the imbalance of spoofing.

    Parameters
    ----------
    i_m : numpy array
        Array containing the imbalance just before a market order.
    dq : numpy array
        Array containing the distribution of dq.
    dp_p : numpy array
        Array containing the distribution of dp.
    weight : numpy array
        Array containing the weight.
    k : int
        Level where the spoofing is performed.
    depth : int
        Depth of the LOB.
    bid_volumes : pandas dataframe
        Dataframe containing the bid volumes.
    ask_volumes : pandas dataframe
        Dataframe containing the ask volumes.
    message : pandas dataframe
        Dataframe containing the message data.
    
    Returns
    -------
    i_spoof : numpy array
        Array containing the imbalance of spoofing.'''

    # Mean of ask_volumes and bid_volumes up to level 4 for each timestamp.
    executions = executions_finder(message)
    H = np.zeros(message.shape[0])
    t = executions['Time'].value_counts().index
    c = executions['Time'].value_counts().values
    d = {'Time': t, 'Count': c}
    df = pd.DataFrame(data=d)
    for i in tqdm(range(df.shape[0]), desc='Computing H'):
        volumes = executions[executions['Time'] == df['Time'][i]]
        volumes = volumes[volumes['Direction'] == -1]['Size']
        for j, i in zip(volumes.index, range(volumes.shape[0])):
            H[j] = volumes.sum() - volumes.iloc[:i].sum()
    a = ask_volumes.iloc[:, :4].mean(axis=1)
    b = bid_volumes.iloc[:, :4].mean(axis=1)
    rho = H / a
    # Fix a value of the level k where I want to spoof and then evaluate
    # mu+, Q_k, v_k
    mu_p = dp_p.mean()
    i_spoof = np.zeros([i_m, depth])
    # for t in range(i_m.shape[0]):
    #     for k in range(depth):
    #         Q_k = dq[k + 1:].sum()
    #         v_k = np.array([(i-k) * dq[k + 1:] for i in range(k + 1, depth)]).sum()
    #         if 2 * rho[t] * weight[k] * mu_p * (1 - i_m[t]) / i_m[t]
    
def direction_of_hlo(message):
    hlo = message[message['Event type'] == 5]
    hlo = hlo['Direction'].unique()
    print(hlo)
    return hlo

def dp_dist(parameters, M, tick_ask_price, bid_volumes, ask_volumes, num_events, f):
    '''This function takes as input the parameters, the number of levels, the number of events, the bid and ask volumes,
    the frequency and returns the (negative) objective function that has to be minimize to estimate the values of
    dp^+ and w (ML estimation).
    
    Parameters
    ----------
    parameters : numpy array
        Array containing the parameters.
    M : int
        Number of levels.
    num_events : int
        Number of events.
    bid_volumes : pandas dataframe
        Dataframe containing the bid volumes.
    ask_volumes : pandas dataframe
        Dataframe containing the ask volumes.
    f : int
        Frequency.
        
    Returns
    -------
    obj_fun : float
        Objective function that has to be minimized.'''

    dp = parameters[:M]
    w = parameters[M:]
    imb, _, _ = avg_imbalance(num_events, bid_volumes, ask_volumes, w, f)
    imb = imb[::f]
    shape, loc, scale = stats.skewnorm.fit(imb)
    obj_fun = 0
    xs = imb.copy()
    xs.sort()
    for i in tqdm(range(imb.shape[0] - 1)):
        if np.abs(ask_prices_fdiff[i]) > int((M-1)/2):
            pass
        else:
            idx = np.where(xs==imb[i])[0][0]
            obj_fun += - np.log((imb[i] * dp[ask_prices_fdiff[i]] + (1 - imb[i]) * dp[-ask_prices_fdiff[i]]) * stats.skewnorm.pdf(xs, shape, loc, scale)[idx])

    return obj_fun/M

def callback(x, M, tick_ask_price, bid_volumes, ask_volumes, N, f, iteration):
    with open(f'opt.txt', 'a', encoding='utf-8') as file:
        file.write(f"\nIteration: {iteration}")
        file.write(f"\nCurrent solution for dp:\n{x[:M]} -> sum = {x[:M].sum()}")
        file.write(f"\nCurrent solution for w:\n{x[M:]} -> sum = {x[M:].sum()}",)
        file.write(f"\nObjective value:\n{dp_dist(x, M, tick_ask_price, bid_volumes, ask_volumes, N, f)}")
        file.write("-------------------------------\n")
    
    if iteration % 10 == 0:
        plt.figure(tight_layout=True, figsize=(7,5))
        plt.bar(list(range(-int((M-1)/2),int((M-1)/2) + 1)), x[:M])
        plt.savefig(f'images/dp_p_{iteration}_{info[0]}_{info[1]}.png')
        plt.close()
    obj_fun.append(dp_dist(x, M, tick_ask_price, bid_volumes, ask_volumes, N, f))

def create_callback(M, tick_ask_price, bid_volumes, ask_volumes, N, f):
    iteration = 0
    def callback_closure(x):
        nonlocal iteration
        iteration += 1
        callback(x, M, tick_ask_price, bid_volumes, ask_volumes, N, f, iteration)
    return callback_closure

def constraint_fun1(x):
    return x[:x.shape[0] - 4].sum() - 1

def constraint_fun2(x):
    return x[x.shape[0] - 4:].sum() - 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script performs the analysis of the LOBSTER data.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-TSLA", "--TSLA", action='store_true')
    parser.add_argument("-MSFT", "--MSFT", action='store_true')
    parser.add_argument("-AMZN", "--AMZN", action='store_true')
    parser.add_argument("-imb", "--imbalance_plot", action='store_true', help='Plot the imbalance distribution')
    parser.add_argument("-pp", "--pre_proc", action='store_true', help='Perform data preprocessing')
    parser.add_argument("-dq", "--dq_dist", action='store_true', help='Compute the distribution of dq')
    parser.add_argument("-dp", "--dp_dist", action='store_true', help='Compute the distribution of dp')
    parser.add_argument("-lob", "--lob_reconstruction", action='store_true', help='Reconstruct the limit order book')
    parser.add_argument("-f", "--freq", action='store_true', help='Compute the optimal frequency of sampling')
    parser.add_argument("-j", "--joint_imbalance", action='store_true', help='Compute the joint imbalance distribution')
    parser.add_argument("-i", "--i_spoof", action='store_true', help='Compute the i-spoofing')

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
    info = [['TSLA', '2015-01-01'], ['MSFT', '2018-04-01'], ['AMZN', '2012-06-21']]
    if args.TSLA:
        info = info[0]
    elif args.MSFT:
        info = info[1]
    elif args.AMZN:
        info = info[2]

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
        elif args.AMZN:
            folder_path = 'data/AMZN_2012-06-21_2012-06-21_10'
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
    # f = int(input('Frequency of sampling: '))
    # orderbook, message = orderbook.iloc[::f].reset_index(drop=True), message.iloc[::f].reset_index(drop=True)
    
    # ret = (orderbook['Ask price 1'] / tick).diff()
    # value, count = np.unique(ret[1:], return_counts=True)
    # x = np.arange(value.min(), value.max()+1)
    # mask = np.in1d(x, value)
    # y = np.zeros_like(x)
    # y[mask] = count
    # y = y / count.sum()
    # plt.figure(tight_layout=True, figsize=(8,5))
    # plt.bar(x, y, color='green', edgecolor='black')
    # plt.title(fr'$r_t$ for {info[0]} from {period[0]} to {period[1]}')
    # plt.xlabel('Ticks deviation')
    # plt.ylabel('Frequency')
    # plt.savefig(f'images/ret_{info[0]}_{period[0]}_{period[1]}_{f}.png')

    ''' As the maximal depth for the distribution dp and dq, we take the difference between 
    the 99.5th and 0.5th quantile to obtain the extreme values between which there is the
    99% of the data. Then we divide this value by 2 to obtain the maximal depth in the directions
    of the bid and ask prices. We are assuming symmetry.'''
    # quantile05 = np.quantile(ret[1:], 0.005)
    # quantile95 = np.quantile(ret[1:], 0.995)
    # depth = int((quantile95 - quantile05) / 2)
    n = orderbook.shape[1]
    bid_prices = np.array(orderbook[[f'Bid price {x}' for x in range(1, int(n/4)+1)]])
    bid_volumes = np.array(orderbook[[f'Bid size {x}' for x in range(1, int(n/4)+1)]])
    ask_prices = np.array(orderbook[[f'Ask price {x}' for x in range(1, int(n/4)+1)]])
    ask_volumes = np.array(orderbook[[f'Ask size {x}' for x in range(1, int(n/4)+1)]])
    m, M = bid_prices.min().min(), ask_prices.max().max()

    if args.MSFT:
        depth = 3
    if args.TSLA:
        depth = 10
    if args.AMZN:
        depth = 6
    
    logging.info('\nData loaded: {}.\nDataframe shape: {}.\nDepth: {}.\nPeriod: {} - {}'.format(info, orderbook.shape, depth, period[0], period[1]))

    if args.lob_reconstruction:
        N = int(input('Number of events to consider: '))
        lob_reconstruction(N, tick, m, M, bid_prices, bid_volumes, ask_prices, ask_volumes)
        os.system(f'rm lob_snapshots/*.jpg')
        lob_video('lob_snapshots', info)

    if args.dq_dist:
        '''
        The distribution that I typically obtain is extremely peaked at 0, with a very long tail.
        This is in accordance to the fact that the typical fraction of MO with a size greater than
        the bests is very small (how markets slowly digests...).
        '''
        var = int(input('Recompute (1) or load (2) the dq?: '))

        if var == 1:
            executions = executions_finder(message)
            # dq = dq_dist(executions, tick, depth)
            dq = dq_dist2(executions, orderbook, depth)
            np.save(f'dq_{info[0]}_{period[0]}_{period[1]}.npy', dq)
        elif var == 2:
            dq = np.load(f'dq_{info[0]}_{period[0]}_{period[1]}.npy')

        value, count = np.unique(dq, return_counts=True)
        x = np.arange(value.min(), value.max()+1)
        mask = np.in1d(x, value)
        y = np.zeros_like(x)
        y[mask] = count
        y = y / count.sum()
        plt.figure(tight_layout=True, figsize=(8,5))
        plt.bar(x, y, color='green', edgecolor='black')
        plt.title(fr'$dq$ for {info[0]} from {period[0]} to {period[1]}')
        plt.xlabel('Ticks deviation')
        plt.ylabel('Frequency')
        plt.savefig(f'images/dq_{info[0]}_{period[0]}_{period[1]}.png')
    
    if args.dp_dist:
        os.system('rm opt.txt')
        N = int(input('Number of events to consider: '))
        f = int(input('Frequency of sampling: '))
        M = int(depth)*2 + 1
        initial_params = np.random.uniform(0, 1, M+4)
        print("Initial parameters:\n", initial_params)
        obj_fun = []
        constraint1 = {'type': 'eq', 'fun': constraint_fun1}
        constraint2 = {'type': 'eq', 'fun': constraint_fun2}
        constraints = [constraint1, constraint2]
        bounds = [(0,1) for i in range(M+4)]
        tick_ask_price = orderbook.values[:,0] / tick
        ask_prices_fsampled = tick_ask_price[::f][:N]
        ask_prices_fdiff = np.diff(ask_prices_fsampled).astype(int)
        res = minimize(dp_dist, initial_params, args=(M, ask_prices_fdiff, bid_volumes, ask_volumes, N, f),  \
                       constraints=constraints, method='SLSQP', bounds=bounds, \
                        callback=create_callback(M, ask_prices_fdiff, bid_volumes, ask_volumes, N, f))
        with open(f'opt.txt', 'a', encoding='utf-8') as file:
            file.write(f"\nFINAL RESULT: {res}")
        print(res)
        print("Initial parameters:\n", initial_params)
        np.save('dp.npy', res.x[:M])
        np.save('ws.npy', res.x[M:])
        plt.figure(tight_layout=True, figsize=(5,5))
        plt.plot(obj_fun)
        plt.title('Objective function')
        plt.figure(tight_layout=True, figsize=(5,5))
        plt.bar(list(range(-int((M-1)/2),int((M-1)/2) + 1)), res.x[:M], color='green', edgecolor='black')
        plt.title(r'$dp^+$')
        plt.savefig(f'images/dp_p_{info[0]}_{info[1]}_{f}_{N}.png')
        plt.show()
    
    if args.imbalance_plot:
        weight = np.array([0.6,0.5,0.2,0.1])
        # weight = np.array([0.1,0.2,0.5,0.6])
        var = int(input('Recompute (1) or load (2) the imbalance?: '))
        boot = int(input('Bootstrap (1) or not (2)?: '))
        if boot == 1:
            bootstrap = True
        else:
            bootstrap = False
        if var == 1:
            N = int(input('Number of events to consider: '))
            imb, parameters, errors = avg_imbalance(N, bid_volumes, ask_volumes, weight, f, bootstrap=bootstrap)
            shape, loc, scale = parameters
            shape_error, loc_error, scale_error = errors
        elif var == 2:
            imb = np.load(f'imbalance_{info[0]}_{info[1]}.npy')
            shape, loc, scale = np.load(f'imbalance_{info[0]}_{info[1]}_{f}_params.npy')
            shape_error, loc_error, scale_error = np.load(f'imbalance_{info[0]}_{info[1]}_{f}_errors.npy')
        x = np.linspace(np.min(imb), np.max(imb), 100)
        plt.figure(tight_layout=True, figsize=(15,5))
        plt.plot(x, stats.skewnorm.pdf(x, shape, loc, scale), 'r-', label=fr'Skew Normal {shape:.2f}$\pm${(2 * shape_error):.2f}, {loc:.2f}$\pm${(2 * loc_error):.2f}, {scale:.2f}$\pm${(2*scale_error):.2f}')
        plt.hist(imb, bins=100, histtype='bar', density=True, edgecolor='black', alpha=0.5, label=f'{weight}')
        plt.xlabel('Imbalance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Imbalance distribution for {info[0]} from {period[0]} to {period[1]}')
        plt.savefig(f'images/imbalance_{info[0]}_{info[1]}_{weight}_{f}.png')
        # weight = np.array([0.1,0.2,0.5,0.6])
        # imb, parameters, errors = avg_imbalance(N, bid_volumes, ask_volumes, weight, f, bootstrap=False)
        # shape, loc, scale = parameters
        # shape_error, loc_error, scale_error = errors
        # x = np.linspace(np.min(imb), np.max(imb), 100)
        # plt.plot(x, stats.skewnorm.pdf(x, shape, loc, scale), 'g-', label=f'Skew Normal {shape:.2f}, {loc:.2f}, {scale:.2f}')
        # plt.hist(imb, bins=100, histtype='bar', density=True, alpha=0.5, label=f'{weight}')
        # plt.legend()
    
    if args.joint_imbalance:
        var = int(input('Recompute (1) or load (2) the imbalance?: '))

        if var == 1:
            N = int(input('Number of events to consider: '))
            weight = np.array([0.6,0.5,0.2,0.1])
            imb, params, errors = avg_imbalance(N, bid_volumes, ask_volumes, weight, f=1, bootstrap=False)
            np.save('imbalance_{info[0]}_{info[1]}.npy', imb)
            i_m, i_p = joint_imbalance(message, imb)
            np.save(f'i_m_{info[0]}_{period[0]}_{period[1]}_{f}.npy', i_m)
            np.save(f'i_p_{info[0]}_{period[0]}_{period[1]}_{f}.npy', i_p)

        elif var == 2:
            i_m, i_p = np.load(f'i_m_{info[0]}_{period[0]}_{period[1]}_{f}.npy'), np.load(f'i_p_{info[0]}_{period[0]}_{period[1]}_{f}.npy')

        plt.figure(tight_layout=True, figsize=(7,7))
        plt.scatter(i_m, i_p, s=15, c='green', edgecolors='black', alpha=0.65)
        plt.title(r'Joint imbalance distribution of $(i_{-}, i_{+})$', fontsize=15)
        plt.xlabel(r'$i_{-}$', fontsize=15)
        plt.ylabel(r'$i_{+}$', fontsize=15)
        # sns.kdeplot(x=i_m, y=i_p, levels=5, colors='r', linewidths=1.5)
        plt.savefig(f'images/joint_imbalance_{info[0]}_{period[0]}_{period[1]}_{f}.png')

        # plot the autocorrelation function of i_m and i_p and show them in the same figure with two plots.
        '''The sequence of i_{-} and i_{+} are correlated, as shown in the autocorrelation function plot.
        This fact negates the idea to use the imbalance just before a market order to detect spoofing, since
        the difference between the imbalance just before and after a market order and the long run
        imbalance can be attributed to market conditions (the ones that generates the correlation).'''
        # What about the correlation between the imbalance and the price change? And with the direction of the trade?
        fig, ax = plt.subplots(2, 1, figsize=(13,7))
        fig.suptitle('Autocorrelation function of $i_{-}$ and $i_{+}$', fontsize=15)
        sm.graphics.tsa.plot_acf(i_m, ax=ax[0], lags=100)
        sm.graphics.tsa.plot_acf(i_p, ax=ax[1], lags=100)
        ax[0].set_title(r'$i_{-}$', fontsize=15)
        ax[1].set_title(r'$i_{+}$', fontsize=15)
        plt.savefig(f'images/autocorrelation_imip_{info[0]}_{period[0]}_{period[1]}_{f}.png')
    
    if args.i_spoof:
        i_m, i_p = np.load('i_m.npy'), np.load('i_p.npy')
        weight = np.array([0.6,0.5,0.2,0.1])
        i_spoofing(i_m, weight, bid_volumes, ask_volumes, message)
        
    if args.freq:
        obj_fun = []
        variance = []
        tick_ask_price = orderbook.values[:,0] / tick
        for f in tqdm(range(1, 100)):
            var = sq_dist_objective(f, tick_ask_price, tick)
            # obj_fun.append(obj)
            variance.append(var)
        obj_fun = (np.array(variance)-2)**2

        fig, ax = plt.subplots(2, 1, figsize=(10,7))
        ax[0].plot((np.array(variance)-2)**2)
        ax[0].set_title('Objective function', fontsize=15)
        ax[1].plot(variance)
        ax[1].set_title('Variance', fontsize=15)
        ax[0].vlines(np.where(obj_fun == obj_fun.min())[0][0], -0.2, obj_fun.max(), linestyles='dashed', label=f'Minimum at f = {np.where(obj_fun == obj_fun.min())[0][0] + 1}')
        plt.legend([f'Minimum at f = {np.where(obj_fun == obj_fun.min())[0][0] + 1}'])
        plt.savefig(f'images/freq_{info[0]}_{info[1]}.png')
        print(f'Minimum at f = {np.where(obj_fun == obj_fun.min())[0][0] + 1}')

        # plt.figure()
        # plt.plot((np.array(variance)-2)**2)


    plt.show()

# Personal notes

# Note:
# - dp e dq sono indipendenti? Non ne capisco il motivo.
# - A cosa serve stimare la frequenza ottimale? Ai fini del rilevamento di spoofing su
# singoli asset meglio tenere tutte le informazioni che si hanno a disposizione, no?
# In fin dei conti non mi interessa fare il confronto fra i diversi asset.
# - E' corretto il modo che ho usato per valutare dp+?