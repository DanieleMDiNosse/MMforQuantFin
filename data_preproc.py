import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm
import argparse
import logging

# ORGANIZATION OF THE DATA (FOR EACH TRADING DAY)
# Message file's columns are as follows:
# Time, Event type, Order ID, Size, Pirce, Direction
# Orderbook file's columns are as follows:
# Ask price 1, Ask size 1, Bid price 1, Bid size 1, Ask price 2, Ask size 2, Bid price 2, Bid size 2, ...
# For TSLA the tick size in 2015 was 0.01 USD (100 in the data)

def data_preproc(path):
    orderbook = pd.read_csv(f'{path}')
    # add a rows at the top of the dataframe to indicate the column names
    orderbook.columns = ['Ask price 1', 'Ask size 1', 'Bid price 1', 'Bid size 1', 'Ask price 2', 
                        'Ask size 2', 'Bid price 2', 'Bid size 2', 'Ask price 3', 'Ask size 3', 'Bid price 3', 
                        'Bid size 3', 'Ask price 4', 'Ask size 4', 'Bid price 4', 'Bid size 4', 'Ask price 5', 
                        'Ask size 5', 'Bid price 5', 'Bid size 5', 'Ask price 6', 'Ask size 6', 'Bid price 6', 
                        'Bid size 6', 'Ask price 7', 'Ask size 7', 'Bid price 7', 'Bid size 7', 'Ask price 8', 
                        'Ask size 8', 'Bid price 8', 'Bid size 8', 'Ask price 9', 'Ask size 9', 'Bid price 9', 
                        'Bid size 9', 'Ask price 10', 'Ask size 10', 'Bid price 10', 'Bid size 10']
    n_columns = orderbook.shape[1]
    # Select the minimum and maximum values of the prices
    m, M = orderbook['Bid price 10'].min(), orderbook['Ask price 10'].max()

    # Select the bid and ask prices and volumes
    bid_prices = orderbook[[f'Bid price {x}' for x in range(1, int(n_columns/4)+1)]]
    bid_volumes = orderbook[[f'Bid size {x}' for x in range(1, int(n_columns/4)+1)]]
    ask_prices = orderbook[[f'Ask price {x}' for x in range(1, int(n_columns/4)+1)]]
    ask_volumes = orderbook[[f'Ask size {x}' for x in range(1, int(n_columns/4)+1)]]

    return m, M, bid_prices, bid_volumes, ask_prices, ask_volumes

def lob_reconstruction(N, tick, m, M, bid_prices, bid_volumes, ask_prices, ask_volumes):
    # p_line = np.arange(m, M+tick, tick)
    # volumes = np.zeros_like(p_line)
    n_columns = bid_prices.shape[1]
    lob_snapshots = []
    for iter in tqdm(range(N)):
        p_line = np.arange(m, M+tick, tick)
        volumes = np.zeros_like(p_line)
        d_ask = {ask_prices.iloc[iter][i]: ask_volumes.iloc[iter][i] for i in range(int(n_columns))}
        d_bid = {bid_prices.iloc[iter][i]: bid_volumes.iloc[iter][i] for i in range(int(n_columns))}
        mid_price = bid_prices['Bid price 1'][iter] + 0.5*(ask_prices['Ask price 1'][iter] - bid_prices['Bid price 1'][iter])

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
        plt.title(f'Limit Order Book {iter}')
        plt.bar(np.arange(p_line.shape[0]), volumes, width=2.5, color=colors)
        plt.vlines(mid_price_loc, 0, volumes.max(), color='black', linestyle='--')
        plt.xlabel('Price')
        plt.ylabel('Volume')
        plt.xticks(tick_positions, tick_labels, rotation=90)
        plt.xlim([min_bid_loc - 10, max_ask_loc + 10])
        plt.savefig(f'lob_snapshot_{iter}.png')
        lob_snapshots.append(imageio.imread(f'lob_snapshot_{iter}.png'))
        plt.clf()
        os.system(f'rm lob_snapshot_{iter}.png')
        plt.close()

    imageio.mimsave('lob_snapshots.gif', lob_snapshots, fps=10)



# lob_snapshots = []
# N=500
# L = int(n_columns/2 + 1)
# p_lines = np.zeros((N, L))
# volumes = np.zeros((N, L))
# for day in tqdm(range(N)):
#     p_lines[day, int((L-1)/2)] = 0.5*(bid_prices.iloc[day, 0] + ask_prices.iloc[day, 0])
#     p_lines[day, :int((L-1)/2)] = bid_prices.iloc[day].sort_values(ascending=True)
#     p_lines[day, int((L-1)/2 + 1):] = ask_prices.iloc[day]

#     volumes[day, :int((L-1)/2)] = bid_volumes.iloc[day].sort_values(ascending=True)
#     volumes[day, int((L-1)/2 + 1):] = ask_volumes.iloc[day]
#     volumes[day, int((L-1)/2)] = 0

#     plt.figure(tight_layout=True, figsize=(9,6))
#     plt.title('Limit Order Book')
#     plt.bar(np.arange(p_lines.shape[1]), volumes[day])
#     plt.xticks(np.arange(p_lines.shape[1]), p_lines[day], rotation=90)
#     plt.xlabel('Price')
#     plt.ylabel('Volume')
#     # plt.show()

#     plt.savefig(f'lob_snapshot_{day}.png')
#     lob_snapshots.append(imageio.imread(f'lob_snapshot_{day}.png'))
#     plt.clf()
#     os.system(f'rm lob_snapshot_{day}.png')
#     plt.close()

# imageio.mimsave('lob_snapshots.gif', lob_snapshots, fps=10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kyle multiperiod model for market microstructure')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-n", "--n_steps", type=int, help='Number of steps. Default 100', default=100)
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    # plt.style.use('ggplot')


    path = 'data/TSLA_2015-01-01_2015-01-31_10/TSLA_2015-01-02_34200000_57600000_orderbook_10.csv'
    m, M, bid_prices, bid_volumes, ask_prices, ask_volumes = data_preproc(path)
    tick = 0.01 * 10000
    N = 100
    lob_reconstruction(N, tick, m, M, bid_prices, bid_volumes, ask_prices, ask_volumes)
