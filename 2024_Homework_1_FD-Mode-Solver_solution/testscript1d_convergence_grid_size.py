''' Tests the convergence when varying the grid size.
'''
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
h             = 0.2
lam           = 0.78
k0            = 2 * np.pi / lam
e_substrate   = 1.5**2
delta_e       = 1.5e-2
num_modes     = 15
w             = 15.0

#grid_sizes = np.logspace(np.log10(2 * w), np.log10(100 * w), 50)
grid_sizes = np.logspace(np.log10(2 * w), np.log10(30 * w), 20)
eff_eps = np.nan * np.zeros((grid_sizes.size, num_modes))
for i, grid_size in enumerate(grid_sizes):
    Nx = np.ceil(int(0.5 * grid_size / h))
    x = np.arange(-Nx, Nx + 1) * h
    prm = e_substrate + delta_e * np.exp(-(x/w)**2)
    start = time.time()
    mode_eps, guided = guided_modes_1DTE(prm, k0, h)
    N = min(num_modes, len(mode_eps))
    eff_eps[i, :N] = mode_eps[:N]
    stop = time.time()
    print("grid size = %8.3fµm, num_points = %3.f, time = %gs" % (grid_size, Nx, stop - start))

# remove modes that do not exist for any grid size
# = remove mode number if eff_eps is NaN for all grid sizes
eff_eps = eff_eps[:, ~np.all(np.isnan(eff_eps), axis=0)]

# calculate relative error to the value obtained at highest resolution
# this should approximate the true error
rel_error = np.abs(eff_eps[:-1, :] / eff_eps[-1, :] - 1.0)

# plot results
fig, ax = plt.subplots()

x = np.log10(grid_sizes[:-1])
for i in range(0, rel_error.shape[1], 2):
    label = "mode %d" % (i+1)
    ax.plot(grid_sizes[:-1], rel_error[:, i], '.--', label=label)
ax.legend(loc="best")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('discretization [µm]')
ax.set_ylabel('relative error of $\\epsilon_{eff}$')

if save_figures:
    plt.savefig('convergence_grid_size.pdf')
plt.show()
