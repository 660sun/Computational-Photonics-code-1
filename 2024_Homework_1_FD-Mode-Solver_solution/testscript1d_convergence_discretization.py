""" Tests the convergence when varying the discretization.
"""
import time
import numpy as np
from matplotlib import pyplot as plt
from Homework_1_solution import guided_modes_1DTE

plt.rcParams.update({
        'figure.figsize': (12/2.54, 9/2.54),
        'figure.subplot.bottom': 0.15,
        'figure.subplot.left': 0.165,
        'figure.subplot.right': 0.95,
        'figure.subplot.top': 0.9,
        'axes.grid': True,
})
plt.close('all')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
save_figures = False

# %%
grid_size     = 120
lam           = 0.78
k0            = 2 * np.pi / lam
e_substrate   = 1.5**2
delta_e       = 1.5e-2
num_modes     = 15
w             = 15.0

# 31 points logarithmically spaced between 10 and 0.053
h = np.logspace(np.log10(10), np.log10(0.053), 31)
eff_eps = np.nan * np.zeros((h.size, num_modes))
for i, hi in enumerate(h):
    Nx = np.ceil(int(0.5 * grid_size / hi))
    x = np.arange(-Nx, Nx+1) * hi
    prm = e_substrate + delta_e * np.exp(-(x/w)**2)
    start = time.time()
    mode_eps, guided = guided_modes_1DTE(prm, k0, hi)
    N = min(num_modes, len(mode_eps))
    eff_eps[i, :N] = mode_eps[:N]
    stop = time.time()
    print("h = %6.3f, num_points = %3.f, time = %gs" % (hi, Nx, stop - start))

# remove modes that do not exist for all grid discretization
# = remove mode number if eff_eps is NaN for at least one grid discretization
eff_eps = eff_eps[:, ~np.any(np.isnan(eff_eps), axis=0)]

# calculate relative error to the value obtained at highest resolution
# this should approximate the true error
rel_error = np.abs(eff_eps[:-1, :] / eff_eps[-1, :] - 1.0)

# fit power law to the linear section of log-log representation
coefficients = np.zeros((rel_error.shape[1], 2))
exclude_large = 6  # number of large h values to exclude from fit
exclude_small = 7  # number of small h values to exclude from fit

x = np.log10(h[exclude_large:-exclude_small-1])
for i in range(rel_error.shape[1]):
    y = np.log10(rel_error[exclude_large:-exclude_small, i])
    coefficients[i, :] = np.polyfit(x, y, 1)

# plot results
fig, ax = plt.subplots()
x = np.log10(h[:-1])
for i in range(0, coefficients.shape[0], 2):
    label = "mode %d, k = %1.2f" % (i+1, coefficients[i, 0])
    ax.plot(h[:-1], rel_error[:, i], '.--', label=label)
    y = 10**np.polyval(coefficients[i, :], x)
    ax.plot(h[:-1], y, ls="-", c=colors[i % len(colors)])
ax.legend(loc="best")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('discretization [µm]')
ax.set_ylabel('relative rel_error of $\\epsilon_{eff}$')

if save_figures:
    plt.savefig('convergence_discretization.pdf')
plt.show()