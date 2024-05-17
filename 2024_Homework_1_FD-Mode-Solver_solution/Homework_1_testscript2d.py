'''Solution to the Homework 1 - FD mode solver.
'''
import numpy as np
from matplotlib import pyplot as plt
from Homework_1_solution import guided_modes_2D, guided_modes_2D_direct
import time
import bluered_dark
plt.rcParams.update({
        'figure.figsize': (12/2.54, 9/2.54),
        'figure.subplot.bottom': 0.15,
        'figure.subplot.left': 0.15,
        'figure.subplot.right': 0.9,
        'figure.subplot.top': 0.9,
        'axes.grid': True,
})
plt.close('all')
save_figures = True

grid_size     = 120
number_points = 301
h             = grid_size/(number_points - 1)
lam           = 0.78
k0            = 2*np.pi/lam
e_substrate   = 2.25
delta_e       = 1.5e-2
w             = 15.0
xx            = np.linspace(-grid_size/2-h,grid_size/2+h,number_points+2)
yy            = np.linspace(-grid_size/2,grid_size/2,number_points)
prm           = e_substrate + delta_e * np.exp(-(xx/w)**2)

[x,y] = np.meshgrid(xx,yy,indexing ='ij')

# make permittivity slightly asymmetric to enforce consistent orientation
# of eigenmodes
prm = e_substrate + delta_e*np.exp(-(x**2 + (1-1e-6)*y**2)/w**2)

plt.figure()
plt.pcolormesh(x, y, prm, zorder=-5, shading = 'nearest')
plt.xlabel('x [µm]')
plt.ylabel('y [µm]')
plt.gca().set_aspect('equal')
cb = plt.colorbar()
cb.set_label('permittivity')
if save_figures:
    plt.savefig('guided_modes_2D_permittivity.pdf')
plt.show()

eff_eps, guided = guided_modes_2D(prm, k0, h, 10)

for i in range(len(eff_eps)):
    f = plt.figure()
    plotfield = guided[i,...].real
    plotfield /= plotfield.flat[np.argmax(np.abs(plotfield))]
    plt.pcolormesh(x, y, plotfield, cmap='bluered_dark',
                   zorder=-5, vmin=-1, vmax=1)
    plt.xlabel('x [µm]')
    plt.ylabel('y [µm]')
    plt.xlim(xx[[0,-1]])
    plt.ylim(yy[[0,-1]])
    plt.gca().set_aspect('equal')
    plt.gca().add_artist(plt.Circle((0, 0), w, facecolor='none',
                         edgecolor='w', lw=0.5, ls='--'))
    plt.text(0, yy[0]*0.9, 'mode number = {0:d} '
             '($\\epsilon_{{eff}}$ = {1:1.6g})'.format(i+1, eff_eps[i].real),
             ha = 'center')
    plt.show()
    if save_figures:
        f.savefig('guided_modes_2D_mode{0:02d}.pdf'.format(i+1))
