from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

""" 5% uncertainty refers to a normal distribution of error 
that has a standard deviation of 5% of the mean value"""

k_mean = 0.001
k_std = 5/100 * k_mean

CA0_mean = 1
CA0_std = 5/100 * CA0_mean

N = 1000

K = np.random.normal(k_mean, k_std, N)
CA0 = np.random.normal(CA0_mean, CA0_std, N)


def ode(X, t, k, Ca0):
    ra = -k * (Ca0 * (1 - X))**2
    dXdt = -ra / Ca0
    return dXdt


x_0 = 0
t = np.linspace(0, 3600)

x = []
for k, Ca0 in zip(K, CA0):
    sol = odeint(ode, x_0, t, args=(k, Ca0))
    x.append(sol[-1, 0])

print(f'Result at 1 hour: {np.average(x):1.3f} +- {np.std(x):1.3f} (1 sigma)')


results = linregress(K, CA0)
print(results)

plt.plot(K, CA0, ".")
plt.xlabel("K")
plt.ylabel("CA0")