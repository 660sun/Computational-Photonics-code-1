'''Solution to the Homework 1 - FD mode solver.
'''
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
save_figures = False

grid_size     = 120
number_points = 601
h             = grid_size/(number_points - 1)
lam           = 0.78
k0            = 2*np.pi/lam
e_substrate   = 2.25
delta_e       = 1.5e-2
w             = 15.0
xx            = np.linspace( -grid_size/2, grid_size/2, number_points )
prm           = e_substrate + delta_e * np.exp(-(xx/w)**2)

plt.figure()
plt.plot(xx, prm, '--')
plt.xlabel('position [µm]')
plt.ylabel('permittivity')
plt.xlim(xx[[0, -1]])
if save_figures:
    plt.savefig('guided_modes_1DTE_permittivity.pdf')
plt.show()

eff_eps, guided = guided_modes_1DTE(prm, k0, h)
print(eff_eps)
for i in range(len(eff_eps)):
    plotfield = guided[:,i]
    ind = np.argmax(np.abs(plotfield))
    plotfield = plotfield / plotfield[ind]*delta_e

    f = plt.figure()
    plt.plot(xx, plotfield, label='mode field')
    plt.plot(xx, prm - e_substrate, '--',
             label='permittivity\nchange')
    plt.legend(frameon=False, loc='upper right')
    plt.xlim(xx[[0, -1]])
    plt.ylim(np.array([-1.2, 1.2])*delta_e)
    plt.text(xx[0]*0.9, -1.05*delta_e, 'mode number = {0:d} '
             '($\\epsilon_{{eff}}$ = {1:1.6g})'.format(i+1, eff_eps[i]))
    plt.xlabel('position [µm]')
    plt.ylabel(' ')
    if save_figures:
        f.savefig('guided_modes_1DTE_mode{0:02d}.pdf'.format(i+1))
    plt.show()
