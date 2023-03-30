'''This code is inspired by the code written by Dale Rosenthal's that can be found at
https://sites.google.com/site/dalerosenthal/teaching/market-microstructure'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import cmath
import logging
import argparse
import imageio
from tqdm import tqdm
import os

def cardano(a, b, c, d):
    '''Compute the roots of a cubic equation using the Cardano's formula. For details see
    https://tinyurl.com/2fmtpw4t '''
    p = b**2 / (3 * a**2) - c / a
    q = 2 * b**3 / (27 * a**3) - b * c / (3 * a**2) + d / a
    Q = cmath.sqrt(p / 3)
    R = cmath.sqrt(q / 2)
    theta = cmath.acos(R / cmath.sqrt(Q**3))
    root_1 = -2 * cmath.sqrt(Q) * cmath.cos(theta / 3) - b / (3 * a)
    root_2 = -2 * cmath.sqrt(Q) * cmath.cos((theta + 2 * cmath.pi) / 3) - b / (3 * a)
    root_3 = -2 * cmath.sqrt(Q) * cmath.cos((theta - 2 * cmath.pi) / 3) - b / (3 * a)
    roots = [root_1, root_2, root_3]
    return sorted(roots, key=lambda x: x.real)

def kyle_sim(Sigma_N, sgm_u, Sgm_0, n_steps):
    '''This function evaluates the parameters of the kyle multiperiod model at each time step
    (at each auction) and the loss function. The idea is to reason backwards, starting from a guess of the final 
    value of Sgm_N (the variance of the transaction price ad time t=T) and then compute the
    corresponding value of Sgm_0*. At the end the loss  |Sgm_0 - Sgm_0*| is computed.'''

    delta_t = 1/n_steps # could also be a vector if trades are irregularly spaced
    Sigma = np.zeros(n_steps+1)
    alpha = np.zeros(n_steps+1)
    delta = np.zeros(n_steps+1)
    lambda_ = np.zeros(n_steps+1)
    beta = np.zeros(n_steps+1)
    Sigma[n_steps] = Sigma_N
    alpha[n_steps] = 0
    delta[n_steps] = 0

    if Sigma_N <= 0:
        diff = 99e99
        return {'sqdiff': diff, 
            'Sigma': Sigma, 
            'alpha': alpha, 
            'delta': delta, 
            'lambda': lambda_, 
            'beta': beta}
    
    lambda_[n_steps] = math.sqrt(Sigma[n_steps])/(sgm_u*math.sqrt(2*delta_t))
    
    for i in range(n_steps, 0, -1):

        # Evaluate in backward fashion the parameters
        beta[i-1] = (1-2*alpha[i]*lambda_[i])/(2*lambda_[i]*(1-alpha[i]*lambda_[i])*delta_t)
        Sigma[i-1] = Sigma[i]/(1-beta[i-1]*lambda_[i]*delta_t)
        alpha[i-1] = 1/(4*lambda_[i]*(1-alpha[i]*lambda_[i]))
        delta[i-1] = delta[i] + alpha[i]*lambda_[i]**2*sgm_u**2*delta_t

        # Setting values for the root finding procedure
        a = alpha[i-1]*sgm_u**2*delta_t/Sigma[i-1]
        b = -sgm_u**2*delta_t/Sigma[i-1]
        c = -alpha[i-1]
        d = 1/2
        # solve lambda cubic -- which has three real roots.
        lambda_[i-1] = cardano(a, b, c, d)[1].real  # pick the middle root
    
    beta[0] = (1-2*alpha[0]*lambda_[0])/(2*lambda_[0]*(1-alpha[0]*lambda_[0])*delta_t)

    diff = (Sigma[0] - Sgm_0)**2
    return {'sqdiff': diff, 
            'Sigma': Sigma, 
            'alpha': alpha, 
            'delta': delta, 
            'lambda': lambda_, 
            'beta': beta}

def loss(Sigma_N, sgm_u, Sgm_0, n_steps):
    '''Wrapper function to be used by the scipy.optimize.minimize function. It returns the loss function
    of the kyle multiperiod model.'''
    res = kyle_sim(Sigma_N, sgm_u, Sgm_0, n_steps)
    diff = res['sqdiff']
    return diff

def volume(beta, lambda_, v, n_steps):
    '''This function simulates the volume of the market maker and
     of the informed trader at each time step (at each auction)'''
    p = np.zeros(shape=n_steps)
    p[0] = 2
    delta_x = np.zeros(shape=n_steps)
    delta_u = np.random.normal(0, sgm_u/np.sqrt(n_steps), size=n_steps)
    delta_t = 1/n_steps

    for n in range(1, n_steps):
        delta_x[n] = beta[n] * (v - p[n-1]) * delta_t
        p[n] = p[n-1] + lambda_[n] * (delta_x[n] + delta_u[n])
    
    return {'delta_x': delta_x, 'delta_u': delta_u, 'p': p}
        


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Kyle multiperiod model for market microstructure')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-n", "--n_steps", type=int, help='Number of steps', default=100)
    parser.add_argument("-su", "--sgm_u", type=int, help='Standard deviation of the volumes of noise traders (constant for each auction)', default=0.9)
    parser.add_argument("-s0", "--Sgm_0", type=int, help='Variance of the initial price (what the MM conjectures) At each auction this quantity will reduce -> the MM learns', default=0.16)
    parser.add_argument("-p0", "--p0", type=int, help='Initial price', default=2)
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    plt.style.use('seaborn')

    n_steps = args.n_steps
    # standard deviation of the volumes of noise traders (constant for each auction)
    sgm_u = args.sgm_u
    # variance of the initial price (what the MM conjectures) At each auction this quantity
    # will reduce -> the MM learns
    Sgm_0 = args.Sgm_0
    p0 = args.p0
    v = np.random.normal(p0, Sgm_0)

    Sgm_N_guess = np.random.uniform(0,1)
    opt = minimize(loss, Sgm_N_guess, args=(sgm_u, Sgm_0, n_steps), method='BFGS')
    print(opt)

    res = kyle_sim(opt.x, sgm_u, Sgm_0, n_steps)
    volumes = volume(res['beta'], res['lambda'], v, n_steps)

    fig, axs = plt.subplots(4,1, tight_layout=True, figsize=(7,8))
    auctions = np.arange(n_steps+1)
    axs[0].set_title(fr'$\sigma_u$ = {sgm_u:.2f} ; $\Sigma_0$ = {Sgm_0:.2f} ; $\Sigma_N$ = {opt.x[0]:.2f}')
    axs[0].scatter(auctions, res['Sigma'], s=15, c='black', alpha=0.8)
    axs[0].set_ylabel(r'$\Sigma_n$')
    axs[1].scatter(auctions, res['lambda'], s=15, c='black', alpha=0.8)
    axs[1].set_ylabel(r'$\lambda$')
    axs[2].scatter(np.arange(n_steps), volumes['delta_x'], label='Informed', s=15, c='black', alpha=0.8)
    axs[2].scatter(np.arange(n_steps), volumes['delta_u'], label='Uninformed', s=15, c='green', alpha=0.8)
    axs[2].set_ylabel(r'$\Delta x$ and $\Delta_u$')
    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True)
    axs[3].plot(volumes['p'], 'k', alpha=0.8)
    axs[3].set_ylabel('p')
    axs[3].set_xlabel('Auctions n')
    axs[3].hlines(v, 0, volumes['p'].shape[0], linestyles='dashed')
    axs[3].text(10, v, f"True Price = {v:.2f}")
    plt.show()

    # Generate a gif of all the subplots above varying the value of sgm_u in the range [0,1]
    
    auctions = np.arange(n_steps+1)
    images = []
    for sgm_u in tqdm(np.linspace(0.5,3,100)):
        fig, axs = plt.subplots(4,1, tight_layout=True, figsize=(9,8))
        res = kyle_sim(opt.x, sgm_u, Sgm_0, n_steps)
        volumes = volume(res['beta'], res['lambda'], v, n_steps)
        axs[0].set_title(fr'$\sigma_u$ = {sgm_u:.2f}')
        axs[0].scatter(auctions, res['Sigma'], s=15, c='black', alpha=0.8)
        axs[0].set_ylabel(r'$\Sigma_n$')
        axs[1].scatter(auctions, res['lambda'], s=15, c='black', alpha=0.8)
        axs[1].set_ylabel(r'$\lambda$')
        axs[2].scatter(np.arange(n_steps), volumes['delta_x'], label='Informed', s=15, c='black', alpha=0.8)
        axs[2].scatter(np.arange(n_steps), volumes['delta_u'], label='Uninformed', s=15, c='green', alpha=0.8)
        axs[2].set_ylabel(r'$\Delta x$ and $\Delta_u$')
        axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True)
        axs[3].plot(volumes['p'], 'k', alpha=0.8)
        axs[3].set_ylabel('p')
        axs[3].set_xlabel('Auctions n')
        axs[3].hlines(v, 0, volumes['p'].shape[0], linestyles='dashed')
        axs[3].text(10, v, f"True Price = {v:.2f}")
        filename = "plot_{:.2f}.png".format(Sgm_N_guess)
        plt.savefig(filename)
        # Add the image to the list
        images.append(imageio.imread(filename))
        # Clear the plot for the next iteration
        plt.clf()
        os.system(f'rm plot_{Sgm_N_guess:.2f}.png')
        plt.close()


    # Save the list of images as a GIF file
    imageio.mimsave('kyle.gif', images, fps=5)

