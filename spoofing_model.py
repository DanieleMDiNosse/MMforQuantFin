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
from scipy.optimize import root, fsolve

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
        Dataframe containing the message data.'''

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

    return orderbook, message

def prices_and_volumes(orderbook, N):
    orderbook = orderbook.iloc[:N]
    n = orderbook.shape[1]
    bid_prices = np.array(orderbook[[f'Bid price {x}' for x in range(1, int(n/4)+1)]])
    bid_volumes = np.array(orderbook[[f'Bid size {x}' for x in range(1, int(n/4)+1)]])
    ask_prices = np.array(orderbook[[f'Ask price {x}' for x in range(1, int(n/4)+1)]])
    ask_volumes = np.array(orderbook[[f'Ask size {x}' for x in range(1, int(n/4)+1)]])
    return bid_prices, bid_volumes, ask_prices, ask_volumes

def executions_finder(message, N, f):
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

    message = message.iloc[f:N]
    executions = message[message['Event type'] == 4] # Select the market orders

    return executions

def dq_dist(executions, orderbook, N, f, depth):
    '''This function takes as input the executions dataframe, the orderbook dataframe, the number of events, the frequency
    and the depth and returns the distribution dq. It considers all the events after the f-th one.
    
    Parameters
    ----------
    executions : pandas dataframe
        Dataframe containing the executions data from the f-th event.
    orderbook : pandas dataframe
        Dataframe containing the orderbook data.
    N : int
        Number of events.
    f : int
        Frequency.
    depth : int
        Depth of the LOB.
    
    Returns
    -------
    dq : numpy array
        Array containing the distribution dq.'''

    orderbook = orderbook.iloc[:N]
    dq = []
    timestamps = executions['Time'].value_counts().index    # Select the timestamps of executions (i.e. MOs)
    counts = executions['Time'].value_counts().values       # Select the number of executions for each timestamp
    data = {'Time': timestamps, 'Count': counts}            # Create a dictionary with the timestamps as keys and the counts as values
    df = pd.DataFrame(data=data)                            # Create a dataframe with the number of executions for each timestamp (i.e. for each MO)

    for i in tqdm(range(df.shape[0]), desc='Computing dq'):
        executions_slice = executions[executions['Time'] == df['Time'][i]]       # Select all the executions for the specific timestamp. 
                                                                                 # Remember that executions is the message file filtered by event type = 4
        total_volume = executions_slice['Size'].sum()                            # Compute the total volume for of the market order

        if executions_slice.index[0] - 1 < 0:
            start = 0
        else:
            start = executions_slice.index[0] - 1

        # Each line of the orderbook contains information about the LOB already updated
        # by the action of the corresponding line in the message file. In other words,
        # each line of the orderbook contains the LOB after the action of the corresponding
        # line in the message file. Hence, I have to select the orderbook slice corresponding
        # to the market order. Specifically, I select the rows of the orderbook dataframe that are
        # between the index just before and just after the executions of the market order.
        orderbook_slice = orderbook.iloc[start: executions_slice.index[-1] + 1]  # Select the orderbook slice corresponding to the market order.
                                                                                 # Specifically, I select the rows of the orderbook dataframe that are
                                                                                 # between the index just before and just after the executions of the market order.

        vol, tick_shift, j = 0, 0, 1
        if executions[executions['Time'] == df['Time'][i]]['Direction'].iloc[0] == 1: # If the direction is 1, the price has decreased and I have to consider the bid side
            while vol <= total_volume and j <= int(orderbook.shape[1]/4):
                vol += orderbook_slice[f'Bid size {j}'].iloc[0]
                j += 1

            # If the volume of the first j-1 levels is just the same as the total volume of the MO,
            # such MO has completely depleted the first j-1 levels of the LOB. Hence, the MO has
            # caused a price deviation equal to j-1 ticks. other wise, the j-1th level has not been
            # completely depleted and the MO has caused a price deviation equal to j-2 ticks.
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

        value, count = np.unique(dq, return_counts=True)
        x = np.arange(-depth, depth+1)
        mask = np.in1d(x, value)
        y = np.zeros_like(x)
        y[mask] = count
        y = y / count.sum()
        np.save(f'output/dq_{info[0]}_{period[0]}_{period[1]}_{N}.npy', y)
    return np.array(y)

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

def optimal_sampling(N, orderbook, tick, target_var=2):
    '''This function takes as input the number of events, the orderbook dataframe, the tick size and the target variance
    and returns the vector of the objective function, the variance of the ask price sampled at each guess f and the final
    optimal sampling frequency.

    The optimal sampling frequency is computed minimizing the square distance betweem
    the empirical variance of the ask price sampled at each f events and the target variance.
    The target variance is set to 2. The optimal frequency f* depends on the number of events
    you want to consider. Hence, you have to compute it every time you change the number of events.
    
    Parameters
    ----------
    N : int
        Number of events.
    orderbook : pandas dataframe
        Dataframe containing the orderbook data.
    tick : float
        Tick size.
        
    Returns
    -------
    f : int
        Optimal sampling frequency.
    '''
    obj_fun = []
    variance = []
    tick_ask_price = (orderbook.values[:,0] / tick)[:N]
    for f in tqdm(range(1, 100)):
        var = sq_dist_objective(f, tick_ask_price)
        variance.append(var)
    obj_fun = (np.array(variance) - target_var)**2
    f = np.where(obj_fun == obj_fun.min())[0][0] + 1
    return obj_fun, variance, f

def compute_support(orderbook, message, f, tick, N):
    '''Once f* is evaluated, I have to compute the support of the distributions dp and dq.
    The support is the maximal depth of the LOB that I want to consider.
    
    As the maximal depth for the distribution dp and dq, we take the difference between 
    the 99.5th and 0.5th quantile to obtain the extreme values between which there is the
    99% of the data. Then we divide this value by 2 to obtain the maximal depth in the directions
    of the bid and ask prices. We are assuming symmetry.'''

    orderbook, message = orderbook.iloc[:N], message.iloc[:N]
    orderbook, message = orderbook.iloc[::f].reset_index(drop=True), message.iloc[::f].reset_index(drop=True)
    ret = (orderbook['Ask price 1'] / tick).diff()
    quantile05 = np.quantile(ret[1:], 0.005)
    quantile95 = np.quantile(ret[1:], 0.995)
    depth = int((quantile95 - quantile05) / 2)
    return depth

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

def joint_imbalance(message, imbalance, N, f):
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
    message = message.iloc[f:N].reset_index(drop=True) # the first element of imbalance is the f-th one.
                                                       # Since I have to take the values of the imbalance just before
                                                       # and after a market order, I have to consider all the market orders
                                                       # executed after the f-th event. With reset_index I reset the index
                                                       # of the message dataframe so that it coincides with the index of
                                                       # imbalance.
    executions = message[message['Event type'] == 4] # Here I actually select the market orders after the f-th event
    executions_m = executions[executions['Direction'] == -1] # Select the market orders initiated by a sell order.
    execution_p = executions[executions['Direction'] == 1] # Select the market orders initiated by a buy order.

    t = executions_m['Time'].value_counts().index
    c = executions_m['Time'].value_counts().values
    d = {'Time': t, 'Count': c}
    df_m = pd.DataFrame(data=d) # Create a dataframe with the number of executions for each timestamp
    print(df_m.shape)

    t = execution_p['Time'].value_counts().index
    c = execution_p['Time'].value_counts().values
    d = {'Time': t, 'Count': c}
    df_p = pd.DataFrame(data=d) # Create a dataframe with the number of executions for each timestamp
    print(df_p.shape)
    
    # Create two arrays containing the imbalance just before and after a market order.
    # For istance, i_m is evaluated considering the last imbalance value just before the
    # beginning of the market order. The index of this last imbalance value is found by
    # selecting the minimum index of the executions dataframe for each timestamp and taking
    # the previous index value. A similar (but specular) procedure is applied to i_p.
    # I subtract f to the index of the executions dataframe because, again, the first value
    # of imbalance is the f-th. The indexes of executions (or in general the message file) and
    # of imbalance are shifted by f.
    i_mm = np.array([imbalance[np.array(executions[executions['Time'] == time].index).min() - 1] for time in tqdm(df_m['Time'], desc='Computing i_mm')])
    i_pm = np.array([imbalance[np.array(executions[executions['Time'] == time].index).max() + 1] for time in tqdm(df_m['Time'], desc='Computing i_pm')])

    i_mp = np.array([imbalance[np.array(execution_p[execution_p['Time'] == time].index).min() - 1] for time in tqdm(df_p['Time'], desc='Computing i_mp')])
    i_pp = np.array([imbalance[np.array(execution_p[execution_p['Time'] == time].index).max() + 1] for time in tqdm(df_p['Time'], desc='Computing i_pp')])

    return i_mm, i_pm, i_mp, i_pp

def implicit_eq_i_spoof(x, i_mm, a, b, w_k, Q_k, rho_t, mu_p, v_k):
    spoof_term = b / (a + b + x * w_k)
    # print('spoof_term: ', spoof_term)
    # print('i_mm: ', i_mm)
    # print('w_k: ', w_k)
    # print('x: ', x)
    implicit_equation = x - 1 - 1 / i_mm - (1 - i_mm) * w_k / Q_k * max(0, 2 * rho_t * w_k * mu_p * (1 - i_mm) / i_mm * spoof_term**2 /
                                                                  - (Q_k * rho_t + v_k))
    return implicit_equation

def MO_volumes(message, N, f):

    executions = executions_finder(message, N, f) # Select the executions from the f-th event
    executions = executions[executions['Direction'] == -1] # Select the market orders initiated by a sell order.
                                                           # This means that an agent has bought causing the price 
                                                           # to increase.
    H = np.zeros(N) # H will be the market order volume at each timestamp of
                                   # a market order execution. It will be something  that is
                                   # zero when there is no market order and then a decreasing
                                   # function when there is a market order.
    t = executions['Time'].value_counts().index # Select the timestamps relative to a market order
    c = executions['Time'].value_counts().values # Select the number of executions for each market order
    d = {'Time': t, 'Count': c}
    df = pd.DataFrame(data=d)
    # idxs contains all the indexes of the message files that are 1-event before every block
    # of market orders. For instance, if there are 3 market orders at time 1, 2 and 3, idxs will
    # contain 0.
    # [executions['Time'] == time] - > Select all the executions at time t=time drawn from df
    # .index.min() - > Select the minimum index of the market order block at time t=time
    # -1 -> Select the index that is 1-event before the first market order of the block
    idxs = np.array([(executions[executions['Time'] == time].index.min() - 1) for time in df['Time']]) # Select all the indexes that are 1-event before the 
                                                                                                                     # first market order of every blocks (of MOs)
    idxs = idxs - f # Subtract f to the indexes because the first value of every vector
                    # that is computed considering a windows of f events has the first
                    # index that is actually the f-th one of the original files.
    for i in tqdm(range(df.shape[0])):
        market_order = executions.iloc[:N][executions['Time'] == df['Time'][i]] # Select the market order from the message file
        volumes = market_order['Size']
        # volumes.index contains the indexes of the message file that correspond to the market order
        for j, k in zip(volumes.index, range(volumes.shape[0])): # Compute the volume of the market order at each timestamp
            H[j] = volumes.iloc[k]

    return H, idxs

def v_spoofing(i_mm, dq, dp_p, H, weight, depth, f, ask_volumes, bid_volumes, idxs, x0):
    '''This function takes as input the imbalance just before a market order, the distribution of dq, the distribution
    of dp, the weight, the level k where the spoofing is performed, the depth, the bid and ask volumes and the message
    dataframe and returns the imbalance of spoofing.

    Parameters
    ----------
    i_mm : numpy array
        Array containing the imbalance just before a sell market order.
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
    
    print('H shape: ', H.shape)
    print('idxs shape: ', idxs.shape)
    print('i_mm shape: ', i_mm.shape)

    rho = np.zeros(idxs.shape[0])
    a = np.zeros(idxs.shape[0])
    b = np.zeros(idxs.shape[0])
    for t, tt in zip(idxs, np.arange(idxs.shape[0])):
        a[tt] = (ask_volumes[t:t-f:-1].sum(axis=0).sum() / (weight.shape[0]*f))
        b[tt] = (bid_volumes[t:t-f:-1].sum(axis=0).sum() / (weight.shape[0]*f))
        rho[tt] = weight.shape[0] * H[t:t-f:-1].sum() / a[tt]
    
    mu_p = sum(value * probability for value, probability in zip(np.arange(-depth, depth+1), dp_p))
    
    v_spoof = np.zeros(shape=[i_mm.shape[0], depth]) # One value of i_spoof for each time and each level k
    Q = np.zeros(shape=depth)
    v = np.zeros(shape=depth)
    for k in tqdm(range(depth), desc='Levels'):
        Q[k] = dq[depth + k + 1:].sum()
        v[k] = np.array([(i-k) * dq[depth + k + 1:] for i in range(depth + k + 1, 2*depth+1)]).sum()
        # t runs over all the times of the market orders
        c0 = 0
        for t in zip(range(i_mm.shape[0])):
            solution = fsolve(implicit_eq_i_spoof, x0=x0, args=(i_mm[t],a[t], b[t], weight[k], Q[k], rho[t], mu_p, v[k]))
            check = max(0, 2 * rho[t] * weight[k] * mu_p * (1 - i_mm[t]) / i_mm[t] * solution**2 - (Q[k] * rho[t] + v[k]))
            if check == 0:
                c0 += 1
            # if solution > i_mm[t]:
            #     i_spoof[t, k] = np.log(-1)
            # else:
            v_spoof[t, k] = solution
        print(c0/i_mm.shape[0])

    return v_spoof, Q, v, a, b, rho, mu_p

def i_spoofing_total(a, b, v_spoof, weights, depth):
    
    i_spoof = np.zeros(shape=[a.shape[0], weights.shape[0]])
    for lev in range(depth):
        spoof_term = (v_spoof[:, lev] * weights[lev]).sum(axis=0)
        i_spoof[:, lev] = b / (a + b + spoof_term)
    return i_spoof

# def v_spoofing(i_spoof, i_mm, w, Q, v, a, idxs, rho, mu_p):
#     v_spoof = np.zeros(shape=[i_spoof.shape[0], w.shape[0]])
#     for k in tqdm(range(w.shape[0]), desc='Levels'):
#         for t, ts in tqdm(zip(range(i_spoof.shape[0]), idxs)):
#             if i_spoof[t, k] == 0:
#                 v_spoof[t, k] = 0
#             else:
#                 v_spoof[t, k] = a[t] / Q[k] * max(0, (2 * rho[t] * w[k] * mu_p * (1 - i_mm[t]) / i_mm[t] * i_spoof[t, k]**2 - (Q[k] * rho[t] + v[k])))
#     return v_spoof

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
    w = parameters[M:] # set the last M values for w (depth + 1)

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

    # Read the data unsliced
    orderbook = pd.read_pickle(f'{info[0]}_orderbook_{info[1]}.pkl')
    message = pd.read_pickle(f'{info[0]}_message_{info[1]}.pkl')
    period = [message['Time'].iloc[0].strftime('%Y-%m-%d'),  message['Time'].iloc[-1].strftime('%Y-%m-%d')]

    # Select the number of events to consider
    N = int(input(f'Number of events to consider (out of {orderbook.shape[0]}): '))

    target_var = 2
    logging.info('\nComputing the optimal sampling frequency...')
    obj_fun, variance, f = optimal_sampling(N, orderbook, tick, target_var)
    fig, ax = plt.subplots(2, 1, figsize=(10,7))
    ax[0].plot((np.array(variance) - target_var)**2, 'k')
    ax[0].set_title('Objective function', fontsize=15)
    ax[0].legend([f'Minimum at f = {f}'])
    ax[0].set_xlabel('Frequency')
    ax[1].plot(variance, 'k')
    ax[1].set_title('Variance', fontsize=15)
    ax[1].set_xlabel('Frequency')
    ax[0].vlines(np.where(obj_fun == obj_fun.min())[0][0], -0.2, obj_fun.max(), linestyle='dashed', color='red', label=f'Minimum at f = {np.where(obj_fun == obj_fun.min())[0][0] + 1}')
    plt.savefig(f'images/freq_{info[0]}_{info[1]}.png')
    
    logging.info('Computing the support for the distributions dp and dq...\n')
    depth = compute_support(orderbook, message, f, tick, N)

    # This vectors start from the very first element of the orderbook
    bid_prices, bid_volumes, ask_prices, ask_volumes = prices_and_volumes(orderbook, N)

    tick_ask_price = orderbook.values[:,0] / tick
    ask_prices_fsampled = tick_ask_price[:N][::f] # ::f means that I take the first element and then I skip f elements and take that one and so on
    ask_prices_fdiff = np.diff(ask_prices_fsampled).astype(int)[1:] # Compute the difference between the ask prices sampled every f events.
                                                                    # Remember that you want to consider the price change between t and t+f
                                                                    # and the value of the imbalance at time t. The first element of ask_prices_fdiff
                                                                    # is the price change between the initial time and the f events after. Since I do not
                                                                    # have the value of the imbalance at the initial time, I have to skip it. Indeed,
                                                                    # the first element of the imbalance is the one computed considering the f events
                                                                    # after the initial time.
    np.save(f'output/ask_prices_fdiff_{info[0]}_{info[1]}', ask_prices_fdiff)

    # These vectors start from the f-th element of the orderbook
    bid_volumes_fsummed, ask_volumes_fsummed = volumes_fsummed(N, f, bid_volumes, ask_volumes)

    logging.info('\nData loaded: {}.\nDataframe shape: {}.\nf: {}.\nDepth: {}.\nPeriod: {} - {}\n'.format(info, orderbook.shape, f, depth, period[0], period[1]))

    if args.lob_reconstruction:
        m, M = bid_prices.min().min(), ask_prices.max().max()
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
            executions = executions_finder(message, N, f) # All the MOs from the f-th event
            dq = dq_dist(executions, orderbook, N, depth)
        elif var == 2:
            dq = np.load(f'dq_{info[0]}_{period[0]}_{period[1]}.npy')
        
        plt.figure(tight_layout=True, figsize=(8,5))
        plt.bar(np.arange(-depth, depth+1), dq, color='green', edgecolor='black')
        plt.title(fr'$dq$ for {info[0]} from {period[0]} to {period[1]}')
        plt.xlabel('Ticks deviation')
        plt.ylabel('Frequency')
        plt.savefig(f'images/dq_{info[0]}_{period[0]}_{period[1]}.png')
    
    if args.dp_dist:
        os.system('rm opt.txt')
        M = int(depth)*2 + 1
        initial_params = np.random.uniform(0, 1, M+depth)
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
            bounds = [(0,1) for i in range(M+depth)]
                                            

        fig, ax = plt.subplots(2, 1, figsize=(8,5), tight_layout=True)
        imb = avg_imbalance_faster(N, bid_volumes_fsummed, ask_volumes_fsummed, np.array([0.6,0.5,0.2,0.1]), f)
        print('imbalanace shape ', N, f, imb.shape)
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
        callback_func = create_callback(M, ask_prices_fdiff, bid_volumes_fsummed, ask_volumes_fsummed, N, f)

        res = minimize(dp_dist, initial_params, args=(M, ask_prices_fdiff, bid_volumes_fsummed, ask_volumes_fsummed, N, f, depth),  \
                    constraints=constraints, method=optimizer, bounds=bounds, \
                    callback=callback_func)

        with open(f'opt.txt', 'a', encoding='utf-8') as file:
            file.write(f"\nFINAL RESULT: {res}")
        print(res.x)
        print("Initial parameters:\n", np.exp(initial_params))
        np.save(f'output/dp_p_{info[0]}_{info[1]}_{f}_{N}.npy', res.x[:M])
        np.save(f'output/ws_{info[0]}_{info[1]}_{f}_{N}.npy', res.x[M:])
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
            imb, avg_imbalance_faster(N, bid_volumes_fsummed, ask_volumes_fsummed, weight, f)
            np.save(f'output/imbalance_{info[0]}_{info[1]}.npy', imb)
            i_mm, i_pm, i_mp, i_pp = joint_imbalance(message, imb, N, f)
            np.save(f'output/i_mm_{info[0]}_{period[0]}_{period[1]}_{f}.npy', i_mm)
            np.save(f'output/i_pm_{info[0]}_{period[0]}_{period[1]}_{f}.npy', i_pm)
            np.save(f'output/i_mp_{info[0]}_{period[0]}_{period[1]}_{f}.npy', i_mp)
            np.save(f'output/i_pp_{info[0]}_{period[0]}_{period[1]}_{f}.npy', i_pp)

        elif var == 2:
            i_mm, i_pm, i_mp, i_pp = np.load(f'i_mm_{info[0]}_{period[0]}_{period[1]}_{f}.npy'), \
                np.load(f'i_pm_{info[0]}_{period[0]}_{period[1]}_{f}.npy'), \
                np.load(f'i_mp_{info[0]}_{period[0]}_{period[1]}_{f}.npy'), \
                np.load(f'i_pp_{info[0]}_{period[0]}_{period[1]}_{f}.npy')

        plt.figure(tight_layout=True, figsize=(7,7))
        plt.scatter(i_mm, i_pm, s=15, c='green', edgecolors='black', alpha=0.65)
        plt.title(r'Joint imbalance distribution of $(i_{-}, i_{+})$ for sell MOs', fontsize=15)
        plt.xlabel(r'$i_{-}$', fontsize=15)
        plt.ylabel(r'$i_{+}$', fontsize=15)
        # sns.kdeplot(x=i_mm, y=i_pm, levels=5, colors='r', linewidths=1.5)
        plt.savefig(f'images/joint_imbalance_{info[0]}_{period[0]}_{period[1]}_{f}.png')

        # plot the autocorrelation function of i_m and i_p and show them in the same figure with two plots.
        '''The sequence of i_{-} and i_{+} are correlated, as shown in the autocorrelation function plot.
        This fact negates the idea to use the imbalance just before a market order to detect spoofing, since
        the difference between the imbalance just before and after a market order and the long run
        imbalance can be attributed to market conditions (the ones that generates the correlation).'''
        # What about the correlation between the imbalance and the price change? And with the direction of the trade?
        fig, ax = plt.subplots(2, 1, figsize=(13,7))
        fig.suptitle('Autocorrelation function of $i_{-}$ and $i_{+}$', fontsize=15)
        sm.graphics.tsa.plot_acf(i_mm, ax=ax[0], lags=100)
        sm.graphics.tsa.plot_acf(i_pm, ax=ax[1], lags=100)
        ax[0].set_title(r'$i_{-}$', fontsize=15)
        ax[1].set_title(r'$i_{+}$', fontsize=15)
        plt.savefig(f'images/autocorrelation_imip_{info[0]}_{period[0]}_{period[1]}_{f}.png')
    
    if args.i_spoof:
        var = int(input('Recompute (1) or load (2) the imbalance, i_m, dq, H_fsummed?:\n '))
        weights = np.load(f'output/ws_{info[0]}_{info[1]}_{f}_{N}.npy') # Load the weights evaluated via the optimization
        if var == 1:
            logging.info('Computing the imbalance...')
            imb = avg_imbalance_faster(N, bid_volumes_fsummed, ask_volumes_fsummed, weights, f)
            logging.info('Computing i_mm...')
            i_mm, _, _, _ = joint_imbalance(message, imb, N, f)
            np.save(f'output/i_mm_{info[0]}_{info[1]}_{f}_{N}.npy', i_mm)
            logging.info('Computing dq...')
            dq = dq_dist(executions_finder(message, N, f), orderbook, N, f, depth)
            np.save(f'output/dq_{info[0]}_{info[1]}_{f}_{N}.npy', dq)
            logging.info('Computing H...')
            H, idxs = MO_volumes(message, N, f)
            np.save(f'output/H_{info[0]}_{info[1]}_{N}_{f}.npy', H)
            np.save(f'output/idxs_{info[0]}_{info[1]}_{N}_{f}.npy', idxs)
            logging.info('\nThe distribution dp+ and the weights w are very costly to compute as they take several hours\
                         to be estimated. Hence, it is better to load the values of dp+ and w. If \
                         you do not have them, run the script with the flag -dp (see -h for more info).')
        if var == 2:
            i_mm = np.load(f'output/i_mm_{info[0]}_{info[1]}_{f}_{N}.npy')
            dq = np.load(f'output/dq_{info[0]}_{info[1]}_{f}_{N}.npy')
            H = np.load(f'output/H_{info[0]}_{info[1]}_{N}_{f}.npy')
            idxs = np.load(f'output/idxs_{info[0]}_{info[1]}_{N}_{f}.npy')

        dp_p = np.load(f'output/dp_p_{info[0]}_{info[1]}_{f}_{N}.npy') # Load the distribution of dp_p evaluated via the optimization

        logging.info('Computing v_spoof_k...')
        for i in range(10):
            x0 = np.random.uniform(0,1)
            v_spoof_k, Q, v, a, b, rho, mu_p = v_spoofing(i_mm, dq, dp_p, H, weights, depth, f,\
                                ask_volumes_fsummed, bid_volumes_fsummed, idxs, x0)
            np.save(f'output/v_spoof_k_{info[0]}_{info[1]}_{f}_{N}.npy', v_spoof_k)

            # Plot the max term
            # fig, ax = plt.subplots(depth,1, figsize=(10,7), tight_layout=True)
            # fig.suptitle(r'$2\rho_t w_k \mu^+ \frac{1-\hat{i}_-(t)}{\hat{i}_-(t)}i_{spoof}^2 - (Q_k \rho_t + \nu_k)$', fontsize=13)
            # term = []
            # for k in range(depth):
            #     for t in range(rho.shape[0]):
            #         y = 2 * rho[t] * weights[k] * mu_p * (1 - i_mm[t]) / i_mm[t] * v_spoof_k[t, k]**2 - (Q[k] * rho[t] + v[k])
            #         term.append(y)
            #     term = np.array(term)
            #     mask = term > 0
            #     y = np.zeros_like(term)
            #     y[np.where(mask)[0]] = term[mask] # np.where(mask) returns the index where mask is True
            #     y[y==0] = np.log(-1)
            #     ax[k].plot(term)
            #     ax[k].scatter(np.arange(rho.shape[0]), y, c='r')
            #     ax[k].set_title(f'Level {k}', fontsize=10)
            #     term = []

            # Plot v_spoof
            fig, ax = plt.subplots(depth, 1, figsize=(10,7), tight_layout=True)
            for i in range(depth):
                # ax[i].plot(i_mm, 'k', alpha=0.5)
                ax[i].plot(v_spoof_k[:,i], 'r')
                ax[i].set_title(rf'$v_spoof_{i+1}$', fontsize=15)
                ax[i].set_xlim(-10, 10000)
                ax[i].set_xlabel('t')
            plt.savefig(f'images/v_spoof_{info[0]}_{info[1]}_{f}_{N}.png')

            # Plot Q, v, rho, mu_p
            # fig, ax = plt.subplots(depth, 1, figsize=(10,7), tight_layout=True)
            # ax[0].scatter(np.arange(Q.shape[0]), Q, c='k')
            # ax[0].set_title(r'$Q_k$', fontsize=15)
            # ax[0].set_xlabel('Level')
            # ax[1].scatter(np.arange(v.shape[0]), v, c='k')
            # ax[1].set_title(r'$v$', fontsize=15)
            # ax[1].set_xlabel('Level')
            # ax[2].plot(rho, 'k')
            # ax[2].set_title(r'$\rho$', fontsize=15)
            # ax[2].hlines(rho.mean(), 0, rho.shape[0], linestyles='dashed', color='red')
            # ax[2].set_xlabel('t')
            # ax[3].hlines(mu_p, 0, N, 'k')
            # ax[3].set_title(r'$\mu_p$', fontsize=15)
            # ax[3].set_xlabel('t')
            # plt.savefig(f'images/Qvrho_t_{info[0]}_{info[1]}_{f}_{N}.png')
            
            # logging.info('Computing v_spoof...')
            # v_spoof = v_spoofing(i_spoof_k, i_mm, weights, Q, v, a, idxs, rho, mu_p)
            # np.save(f'output/v_spoof_{info[0]}_{info[1]}_{f}_{N}.npy', v_spoof)

            # # Plot v_spoof
            # fig, ax = plt.subplots(depth, 1, figsize=(10,7), tight_layout=True)
            # mask = [v_spoof[:, i] > 0 for i in range(4)]
            # for i in range(4):
            #     y = np.zeros_like(v_spoof[:,i])
            #     y[np.where(mask[i])[0]] = np.log(-1)
            #     ax[i].plot(v_spoof[:,i], 'g')
            #     ax[i].scatter(np.arange(v_spoof[:,i].shape[0]), y, c='r')
            #     ax[i].set_title(rf'$v_spoof_{i+1}$', fontsize=15)
            #     ax[i].set_xlim(-10, 10000)
            #     ax[i].set_xlabel('t')
            #     v_spoof_zero = np.zeros(shape=v_spoof.shape[0])
            # plt.savefig(f'images/v_spoof_{info[0]}_{info[1]}_{f}_{N}.png')

            logging.info('Computing i_spoofing...')
            i_spoof_k = i_spoofing_total(a, b, v_spoof_k, weights, depth)
            np.save(f'output/i_spoof_{info[0]}_{info[1]}_{f}_{N}.npy', i_spoof_k)

            # Plot i_spoof_k
            fig, ax = plt.subplots(depth, 1, figsize=(10,7), tight_layout=True)
            for i in range(depth):
                # ax[i].plot(i_mm, 'k', alpha=0.5)
                ax[i].plot(i_spoof_k[:,i], 'r')
                ax[i].set_title(rf'$i_spoof_{i+1}$', fontsize=15)
                ax[i].set_xlim(-10, 10000)
                ax[i].set_xlabel('t')
            plt.savefig(f'images/i_spoof_k_{info[0]}_{info[1]}_{f}_{N}.png')

            # Plot i_spoof(i_mm)
            fig, ax = plt.subplots(depth, 1, figsize=(10,7), tight_layout=True)
            for i in range(depth):
                ax[i].plot(np.linspace(0,1,1000), np.linspace(0,1,1000), 'k', linestyle='dashed', alpha=0.7)
                ax[i].scatter(i_mm, i_spoof_k[:,i], c='g')
                ax[i].set_title(r'$i_{spoof}(i_{-})$', fontsize=15)
                ax[i].set_xlabel(r'$i_{-}$', fontsize=15)
                ax[i].set_ylabel(r'$i_{spoof}$', fontsize=15)
            plt.show()
        # plt.savefig(f'images/i_spoof_imb_{info[0]}_{info[1]}_{f}_{N}.png')

# Figure 2 of the paper
# i = np.random.uniform(0, 1, size=1000)
# dq = np.array([0.006 for i in range(2*depth+1)])
# Q_k = [dq[depth + k +1:].sum() for k in range(depth)]
# v_k = [np.array([(i-k) * dq[depth + k + 1:] for i in range(depth + k + 1, 2*depth+1)]).sum() for k in range(depth)]
# w_k = [0.5,0.5,0.5,0.5]
# rho_t = 3
# mu_p = 3

# for Q, v, w in zip(Q_k, v_k, w_k):
#     solution = []
#     for ii in i_mm:
#         x_initial = 0.5  # Example initial value, you can change it accordingly
#         result = root(implicit_eq_i_spoof, x_initial, args=(ii, w, Q, rho_t, mu_p, v))
#         if result.success:
#             solution.append(result.x)
#         else:
#             solution.append(np.nan)

#     plt.plot(i_mm, solution, label=f"Q_k={Q}, v_k={v}")

#     plt.xlabel('i')
#     plt.ylabel('Solution')
#     plt.title('Roots of the Implicit Equation')
#     plt.legend()
#     plt.grid(True)
# plt.show()

    plt.show()
    print('Last episode of One Piece was amazing!')