import numpy as np 
import matplotlib.pyplot as plt

g = np.loadtxt("coherent.dat")
n = np.arange(len(g))
mean = np.average(g)
stddev = np.sqrt(np.var(g))
fig, ax = plt.subplots()
righty = ax.secondary_yaxis('right')
righty.set_yticks([mean-stddev, mean, mean+stddev])
righty.set_yticklabels([r"$\mu-\sigma$", r"$\mu$", r"$\mu+\sigma$"])

ax.scatter(n, g, color='k', marker='.')
ax.plot([-5, 205],[mean, mean], 'k-')
ax.plot([-5, 205],[mean-stddev, mean-stddev], 'k--')
ax.plot([-5, 205],[mean+stddev, mean+stddev], 'k--')
ax.set_ylabel(r"Conductance / $(2e^2/h)$", size=14)
ax.set_xlabel(r"Disorder configuration", size=14)
ax.grid(True)
ax.set_xlim([-2, 202])

plt.show()