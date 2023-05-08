''' Description: This file contains the code for the valuation of the spread
according to the Glosten-Milgrom model'''
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

def RHS(w, phi, sgm):
    return (s - phi * (2*sgm + s) * np.exp(s/w)) / (phi * np.exp(s/w) + (1 - phi) * np.exp(s/sgm))



# Parameters
phi = 0.1
sgm = 0.1
w = np.linspace(0.1, 2, 1000)
f = [RHS(x, phi, sgm) for x in w]
plt.plot(w, f)
plt.show()

spread = []
for w in tqdm(np.linspace(0.1,2, 1000)):

    s = np.linspace(0.01, 1, 10000)
    lhs = s
    rhs = RHS(s)
    spread.append(s[np.where(np.abs(lhs - rhs) < 0.00001)])

spread = (np.array(spread)).flatten()
print(spread)
plt.scatter(np.arange(50),spread)
plt.show()
s = np.linspace(0.01, 100, 100000)
lhs = s
print(s[np.where(np.abs(lhs - RHS(s)) < 0.00001)])
plt.figure()
plt.plot(s, RHS(s), label='right hand side')
plt.plot(s, lhs, label='left hand side')
plt.xlabel('s')
plt.legend()
plt.show()
