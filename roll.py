import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acovf
plt.style.use('seaborn')

# Parameters
T = 20000
c = 0.01
m0 = 50 # first value of the efficient price
su = 0.01 # standard deviation of the efficient price

m = np.empty(shape=T)
m[0] = m0
p = np.empty(shape=T)

# Model simulation
q = 2 * np.random.binomial(n=1,p=0.5,size=T) - 1 # generation of the arrivals of trades
for i in range(T-1):
    m[i+1] = m[i] + np.random.normal(0,su)
p = m + q * c
plt.figure()
plt.plot(p, label='Transaction price')
plt.plot(m, label='Efficient price')
plt.xlabel('Trading time')
plt.legend()


# Signature plot
# remember that for a diffusion the signature plot is flat
T = 100
tau = np.arange(1,T)
C_emp = np.empty(T)
for t in tau:
    C_emp[t] = np.mean((p[t:] - p[:-t])**2) / t
C_teo = su**2 + 2*c**2/tau

plt.figure()
plt.scatter(tau,C_emp[1:], s=15, c='black', label=r'Empirical $C(\tau)$')
plt.plot(C_teo, label=r'Teorethical $C(\tau)$')
plt.title('Signature plot')
plt.xlabel(r'$\tau$')
plt.legend()

# Autocorrelation function. Remember that from the plot it is possible to estimate
# the value of the spread -> sqrt(-acf[1])
diff = np.diff(p)
sm.graphics.tsa.plot_acf(diff, lags=10)
print(f'Real spread: {2*c}')
print(f'Estimated spread: {2*np.sqrt(-acovf(diff, nlag=10)[1])}')
plt.title('Autocorellation function')

# Estimating the filtered state estimate
# Filtering involves using the model to remove 
# the noise from the data and reveal the underlying signal
# In the case of a MA(1), the "state" of the system is simply the value 
# of the previous error term e_{t-1}. Since the error terms are 
# not directly observed, the state must be estimated using the observed data

mod = sm.tsa.arima.ARIMA(diff, order=(0,0,1))
res = mod.fit()
theta_hat = res.params[1]

eps = np.zeros_like(p)
for t in range(p.shape[0]-1):
    eps[t+1] = p[t+1] - p[t] - theta_hat * eps[t]

# Now, the forecast of the price is the following one, that turns out to be
# the filetered state estimate of the efficient price
f = p + theta_hat * eps

plt.figure()
plt.plot(f, label='filtered state?')
plt.plot(p, label='real')
plt.legend()
plt.show()
