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

IMPORTANTE NOTE
---------------
ALL YOU HAVE DONE IS INTENDED FOR AN AGENT THAT WANTS TO BUY A CERTAIN AMOUNT OF SHARES. SHE
TRIES TO SPOOF THE MARKET BY PLACING LIMIT ORDERS AT SOME PRICE LEVEL AT THE ASK SIDE, IN ORDER
TO CREATE A DOWNSIDE PRESSURE AND TAKE ADVANTAGE OF THE DECREASING PRICE.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import argparse
import logging
from scipy.optimize import minimize, LinearConstraint
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

def dq_dist(executions, orderbook, depth):
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

def sq_dist_objective(f, tick_ask_price):
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

def volumes_fsummed(N, f, bid_volumes, ask_volumes):
    '''This function takes as input the number of events, the frequency, the bid and ask volumes and returns
    the bid and ask volumes summed over f events.
    
    Parameters
    ----------
    N : int
        Number of events.
    f : int
        Frequency.
    bid_volumes : pandas dataframe
        Dataframe containing the bid volumes.
    ask_volumes : pandas dataframe
        Dataframe containing the ask volumes.
    
    Returns
    -------
    bid_volumes_fsummed : numpy array
        Array containing the bid volumes summed over f events.
    ask_volumes_fsummed : numpy array
        Array containing the ask volumes summed over f events.'''

    bid_volumes_fsummed = np.array([bid_volumes[i:i+f].sum(axis=0) for i in range(0, N-f)])
    ask_volumes_fsummed = np.array([ask_volumes[i:i+f].sum(axis=0) for i in range(0, N-f)])

    return bid_volumes_fsummed, ask_volumes_fsummed

@nb.njit
def avg_imbalance_faster(num_events, bid_volumes_fsummed, ask_volumes_fsummed, weight, f):
    '''This function is a faster version of avg_imbalance. The speed up is made possible by the numba library.
    In the original function, some non supported numba functions are used. This version is not able to compute
    the errors on the estimated parameters and is not able to save the output. It is mainly used to compute the
    distribution dp+ and w.'''
    imb = []
    for i in range(0, num_events - f):
        num = dot_product(bid_volumes_fsummed[i], weight)
        det = dot_product((bid_volumes_fsummed[i] + ask_volumes_fsummed[i]), weight)
        imb.append(num/det)
    imb = np.array(imb)
    return imb

def avg_imbalance(num_events, bid_volumes, ask_volumes, weight, f, bootstrap=False, save_output=False):
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
    for i in tqdm(range(0, num_events - f), desc='Computing average imbalance'):
        num = dot_product(bid_volumes[i:i+f, :lev].sum(axis=0), weight)
        det = dot_product((bid_volumes[i:i+f, :lev].sum(axis=0) + ask_volumes[i:i+f, :lev].sum(axis=0)), weight)
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

        if save_output:
            np.save(f'output/imbalance_{info[0]}_{info[1]}_errors', np.array([shape_error, loc_error, scale_error]))
    
    else:
        if save_output:
            np.save(f'output/imbalance_{info[0]}_{info[1]}', imb)
            np.save(f'output/imbalance_{info[0]}_{info[1]}_params', np.array([shape, loc, scale]))
        shape_error, loc_error, scale_error = 0, 0, 0

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

def i_spoofing(i_m, dq, dp_p, weight, k, depth, f,  bid_volumes_fsummed, ask_volumes_fsummed, ask_prices_fdiff, message):
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
    executions = executions_finder(message)   # Select the market orders
    executions[executions['Direction'] == -1] # Select the market orders initiated by a sell order.
                                              # This means that an agent has bought causing the price 
                                              # to increase.
    H = np.zeros(message.shape[0]) # H will be the market order volume at each timestamp of
                                   # a market order execution. It will be something  that is
                                   # zero when there is no market order and then a decreasing
                                   # function when there is a market order.
    t = executions['Time'].value_counts().index # Select the timestamps relative to a market order
    c = executions['Time'].value_counts().values # Select the number of executions for each market order
    d = {'Time': t, 'Count': c}
    df = pd.DataFrame(data=d)

    for i in tqdm(range(df.shape[0]), desc='Computing H'):
        market_order = executions[executions['Time'] == df['Time'][i]] # Select the market order
        volumes = market_order['Size']
        for j, i in zip(volumes.index, range(volumes.shape[0])): # Compute the volume of the market order at each timestamp
            H[j] = volumes.sum() - volumes.iloc[:i].sum()
    
    H_fsummed = np.array([H[i:i+f].sum() for i in range(H.shape[0] - f)]) # Sum the market order volumes over f events

    a = ask_volumes_fsummed.iloc[:, :4].sum() / (N*f) # Mean of ask_volumes_fsummed up to level 4 for each timestamp.
    b = bid_volumes_fsummed.iloc[:, :4].sum() / (N*f) # Mean of bid_volumes_fsummed up to level 4 for each timestamp.
    rho = H_fsummed / a  # "Ratio of the shares to purchase to the depth of the LOB"
    # Fix a value of the level k where I want to spoof and then evaluate
    # mu+, Q_k, v_k
    mu_p = np.array([ask_prices_fdiff[i] * dp_p[ask_prices_fdiff[i] + depth] for i in range(ask_prices_fdiff.shape[0])]).sum()
    i_spoof = np.zeros([i_m, depth])
    # for t in range(i_m.shape[0]):
    #     for k in range(depth):
    #         Q_k = dq[k + 1:].sum()
    #         v_k = np.array([(i-k) * dq[k + 1:] for i in range(k + 1, depth)]).sum()
    #         if 2 * rho[t] * weight[k] * mu_p * (1 - i_m[t]) / i_m[t]

def dp_dist(parameters, M, ask_prices_fdiff, bid_volumes_fsummed, ask_volumes_fsummed, num_events, f, depth):
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

    dp = parameters[:M] # set the first M values for dp (depth*2 + 1)
    w = parameters[M:] # set the last M values for w (4 by default)

    imb = avg_imbalance_faster(num_events, bid_volumes_fsummed, ask_volumes_fsummed, w, f) # evaluate the imbalance
    shape, loc, scale = stats.skewnorm.fit(imb) # compute the parameters of the fitted skew 
                                                # normal distribution.
    imb = imb[::f] # sample the imbalance every f events. Note that the imbalance is
                   # computed averaging the f previous events every time. It starts
                   # from the f-th event and goes on until num_events (N). Hence, it has
                   # one value for every events in the LOB, starting from the f-th one.
                   # In order to have the pair (x_m, i_m) where x_m represents the price
                   # change "influenced" by i_m (so I see i_m at time t and check the price
                   # change at time t+f). 
    obj_fun = 0
    for i in range(imb.shape[0] - 1):
        if np.abs(ask_prices_fdiff[i]) > int((M-1)/2): # if the difference exceeds the max depth, ignore it
            pass
        else:
            # x, dx = np.linspace(imb[i]-0.005, imb[i]+0.005, 100, retstep=True)
            # p_imb = (stats.skewnorm.pdf(x, shape, loc, scale)*dx).sum() # compute the probability of the imbalance. Remember that
                                                                        # the probability is the area under the curve (pdf is the
                                                                        # derivative of the cumulative distribution functoion). 
                                                                        # Hence, I have to multiply the pdf by the step dx and then 
                                                                        # sum over all the values of x within a specified range.
            dp_pp = dp[ask_prices_fdiff[i] + depth]
            dp_pm = dp[-ask_prices_fdiff[i] + depth]
            p_im = stats.skewnorm.pdf(imb[i], shape, loc, scale)

            obj_fun += - np.log((imb[i] * dp_pp + (1 - imb[i]) * dp_pm) * p_im) # compute the objective function (negative log-likelihood)
    
    return obj_fun/imb.shape[0]

def callback(x, M, ask_prices_fdiff, bid_volumes, ask_volumes, N, f, iteration):
    print(f"Iteration: {iteration}")
    with open(f'opt.txt', 'a', encoding='utf-8') as file:
        file.write(f"\nIteration: {iteration}")
        file.write(f"\nCurrent solution for dp:\n{x[:M]} -> sum = {x[:M].sum()}")
        file.write(f"\nCurrent solution for w:\n{x[M:]} -> sum = {x[M:].sum()}")
        file.write(f"\nObjective value:\n{dp_dist(x, M, ask_prices_fdiff, bid_volumes, ask_volumes, N, f, depth)}")
        file.write("-------------------------------\n")
    
    if iteration % 3 == 0:
        plt.figure(tight_layout=True, figsize=(7,5))
        plt.bar(list(range(-int((M-1)/2),int((M-1)/2) + 1)), x[:M])
        plt.savefig(f'images/dp_p_{iteration}_{info[0]}_{info[1]}.png')
        plt.close()
    obj_fun.append(dp_dist(x, M, ask_prices_fdiff, bid_volumes, ask_volumes, N, f, depth))

def create_callback(M, ask_prices_fdiff, bid_volumes, ask_volumes, N, f):
    iteration = 0
    def callback_closure(x, *args):
        if len(args) == 1:
            state = args[0]
        nonlocal iteration
        iteration += 1
        callback(x, M, ask_prices_fdiff, bid_volumes, ask_volumes, N, f, iteration)
    return callback_closure

def constraint_fun1(x):
    return x[:x.shape[0] - 4].sum() - 1

def constraint_fun2(x):
    return x[x.shape[0] - 4:].sum() - 1

def insert_zeros_between_values(v, k):
    new_array = []
    for i in range(len(v)):
        new_array.append(v[i])
        for j in range(k):
            new_array.append(np.log(-1))
    return np.array(new_array[:-k])


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

    '''The optimal sampling frequency is computed minimizing the square distance betweem
    the empirical variance of the ask price sampled at each f events and the target variance.
    The target variance is set to 2. The optimal frequency f* depends on the number of events
    I want to consider. Hence, I have to compute it every time I change the number of events.'''

    logging.info('Computing optimal frequency of sampling')
    N = int(input(f'Number of events to consider (out of {orderbook.shape[0]}): '))
    obj_fun = []
    variance = []
    tick_ask_price = (orderbook.values[:,0] / tick)[:N]
    for f in tqdm(range(1, 100)):
        var = sq_dist_objective(f, tick_ask_price)
        variance.append(var)
    obj_fun = (np.array(variance)-2)**2

    fig, ax = plt.subplots(2, 1, figsize=(10,7))
    ax[0].plot((np.array(variance)-2)**2, 'k')
    ax[0].set_title('Objective function', fontsize=15)
    ax[0].legend([f'Minimum at f = {np.where(obj_fun == obj_fun.min())[0][0] + 1}'])

    ax[1].plot(variance, 'k')
    ax[1].set_title('Variance', fontsize=15)
    ax[0].vlines(np.where(obj_fun == obj_fun.min())[0][0], -0.2, obj_fun.max(), linestyles='dashed', color='red', label=f'Minimum at f = {np.where(obj_fun == obj_fun.min())[0][0] + 1}')
    plt.savefig(f'images/freq_{info[0]}_{info[1]}.png')

    f = np.where(obj_fun == obj_fun.min())[0][0] + 1
    
    '''Once f* is evaluated, I have to compute the support of the distributions dp and dq.
    The support is the maximal depth of the LOB that I want to consider.
    
    As the maximal depth for the distribution dp and dq, we take the difference between 
    the 99.5th and 0.5th quantile to obtain the extreme values between which there is the
    99% of the data. Then we divide this value by 2 to obtain the maximal depth in the directions
    of the bid and ask prices. We are assuming symmetry.'''

    orderbook, message = orderbook.iloc[::f].reset_index(drop=True), message.iloc[::f].reset_index(drop=True)
    ret = (orderbook['Ask price 1'] / tick).diff()
    quantile05 = np.quantile(ret[1:], 0.005)
    quantile95 = np.quantile(ret[1:], 0.995)
    depth = int((quantile95 - quantile05) / 2)

    orderbook = pd.read_pickle(f'{info[0]}_orderbook_{info[1]}.pkl') # Re-load the original files
    message = pd.read_pickle(f'{info[0]}_message_{info[1]}.pkl')
    n = orderbook.shape[1]
    bid_prices = np.array(orderbook[[f'Bid price {x}' for x in range(1, int(n/4)+1)]])
    bid_volumes = np.array(orderbook[[f'Bid size {x}' for x in range(1, int(n/4)+1)]])
    ask_prices = np.array(orderbook[[f'Ask price {x}' for x in range(1, int(n/4)+1)]])
    ask_volumes = np.array(orderbook[[f'Ask size {x}' for x in range(1, int(n/4)+1)]])
    m, M = bid_prices.min().min(), ask_prices.max().max()

    tick_ask_price = orderbook.values[:,0] / tick
    ask_prices_fsampled = tick_ask_price[:N][::f] # ::f means that I take the first element and then I skip f elements and take that one and so on
    ask_prices_fdiff = np.diff(ask_prices_fsampled).astype(int)[1:] # Compute the difference between the ask prices sampled every f events.
                                                                    # Remember that you want to consider the price change between t and t+f
                                                                    # and the value of the imbalance at time t. The first element of ask_prices_fdiff
                                                                    # is the price change between the initial time and the f events after. Since I do not
                                                                    # have the value of the imbalance at the initial time, I have to skip it. Indeed,
                                                                    # the first element of the imbalance is the one computed considering the f events
                                                                    # after the initial time.
    bid_volumes_fsummed, ask_volumes_fsummed = volumes_fsummed(N, f, bid_volumes, ask_volumes)

    logging.info('\nData loaded: {}.\nDataframe shape: {}.\nf: {}.\nDepth: {}.\nPeriod: {} - {}'.format(info, orderbook.shape, f, depth, period[0], period[1]))
    print()

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
            dq = dq_dist(executions, orderbook, depth)
            np.save(f'output/dq_{info[0]}_{period[0]}_{period[1]}.npy', dq)
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
        M = int(depth)*2 + 1
        initial_params = np.random.uniform(0, 1, M+4)
        # initial_params[:M] = np.exp(initial_params[:M]) / np.exp(initial_params[:M]).sum()
        # initial_params[M:] = initial_params[M:] / initial_params[M:].sum()
        # logging.info(f"\nInitial parameters:\n {initial_params}\nSum of all dp's:\n{initial_params[:M].sum()}\nSum of all w's:\n{initial_params[M:].sum()}")
        obj_fun = []

        optimizer = int(input('Choose optimizer (1: trust-constr, 2: SLSQP)\n'))
        if optimizer == 1:
            optimizer = 'trust-constr'
            A1 = np.append(np.ones(M), np.zeros(4))
            A2 = np.append(np.zeros(M), np.ones(4))
            b = np.array([1])
            constraint1 = LinearConstraint(A1, b, b)
            constraint2 = LinearConstraint(A2, b, b)
            constraints = [constraint1, constraint2]
            bounds = [(0,1) for i in range(M+4)]
        elif optimizer == 2:
            optimizer = 'SLSQP'
            constraint1 = {'type': 'eq', 'fun': constraint_fun1}
            constraint2 = {'type': 'eq', 'fun': constraint_fun2}
            constraints = [constraint1, constraint2]
            bounds = [(0,1) for i in range(M+4)]
                                            

        fig, ax = plt.subplots(2, 1, figsize=(8,5), tight_layout=True)
        imb = avg_imbalance_faster(N, bid_volumes_fsummed, ask_volumes_fsummed, np.array([0.6,0.5,0.2,0.1]), f)
        imb = np.append(np.zeros(f), imb)
        ax[0].plot(imb, 'k')
        ax[0].vlines(np.arange(0, imb.shape[0])[f:][::f], 0, 1, linestyles='dashed', color='red')
        ax[0].set_title('Imbalance (with arbitrary weights)')
        ax[1].plot(tick_ask_price[:N], 'k')
        ax[1].vlines(np.arange(f, tick_ask_price[:N].shape[0])[::f], tick_ask_price[:N].min(), tick_ask_price[:N].max(), linestyles='dashed', color='red')
        ax[1].set_title('Ask price')
        ax1 = ax[1].twinx()
        ask_prices_fdiff_extended = insert_zeros_between_values(ask_prices_fdiff, f-1)
        ax1.scatter(np.arange(f, ask_prices_fdiff_extended.shape[0]), ask_prices_fdiff_extended[:-f], color='red', s=15)
        ax[0].set_xlim(-10, 500)
        ax[1].set_xlim(-10, 500)
        plt.show()

        value, count = np.unique(ask_prices_fdiff, return_counts=True)
        x = np.arange(value.min(), value.max()+1)
        mask = np.in1d(x, value)
        y = np.zeros_like(x)
        y[mask] = count
        y = y / count.sum()
        plt.figure(tight_layout=True, figsize=(8,5))
        plt.bar(x, y, color='green', edgecolor='black')
        plt.title(f'Price change sampled every {f} events')
        callback_func = create_callback(M, ask_prices_fdiff, bid_volumes, ask_volumes, N, f)

        res = minimize(dp_dist, initial_params, args=(M, ask_prices_fdiff, bid_volumes, ask_volumes, N, f, depth),  \
                    constraints=constraints, method=optimizer, bounds=bounds, \
                    callback=callback_func)

        with open(f'opt.txt', 'a', encoding='utf-8') as file:
            file.write(f"\nFINAL RESULT: {res}")
        print(res.x)
        print("Initial parameters:\n", np.exp(initial_params))
        np.save('output/dp.npy', res.x[:M])
        np.save('output/ws.npy', res.x[M:])
        plt.figure(tight_layout=True, figsize=(5,5))
        plt.plot(obj_fun)
        plt.title('Objective function')
        plt.figure(tight_layout=True, figsize=(5,5))
        plt.bar(list(range(-int((M-1)/2),int((M-1)/2) + 1)), res.x[:M], color='green', edgecolor='black')
        plt.title(r'$dp^+$')
        plt.savefig(f'images/dp_p_{info[0]}_{info[1]}_{f}_{N}_{optimizer}.png')
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
    
    if args.joint_imbalance:
        var = int(input('Recompute (1) or load (2) the imbalance?: '))

        if var == 1:
            weight = np.array([0.6,0.5,0.2,0.1])
            imb, params, errors = avg_imbalance(N, bid_volumes, ask_volumes, weight, f=1, bootstrap=False)
            np.save('output/imbalance_{info[0]}_{info[1]}.npy', imb)
            i_m, i_p = joint_imbalance(message, imb)
            np.save(f'output/i_m_{info[0]}_{period[0]}_{period[1]}_{f}.npy', i_m)
            np.save(f'output/i_p_{info[0]}_{period[0]}_{period[1]}_{f}.npy', i_p)

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


    plt.show()