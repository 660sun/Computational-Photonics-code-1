'''Homework 1, Computational Photonics, SS 2024:  FD mode solver. 1+1=2D
'''
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def guided_modes_2D(prm, k0, h, numb):
    """Computes the effective permittivity of a quasi-TE polarized guided 
    eigenmode. All dimensions are in µm.
    
    Parameters
    ----------
    prm  : 2d-array
        Dielectric permittivity in the xy-plane
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization
    numb : int
        Number of eigenmodes to be calculated
    
    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated eigenmodes
    guided : 2d-array
        Field distributions of the guided eigenmodes
    """
    dt = np.common_type(np.array([prm]))
    M  = np.zeros((len(prm), len(prm)), dtype = dt)
    for i  in range(len(prm)):
        M[i][i] =  -4/(h**2) + (k0**2) * prm[i][i]
        if i > 0 & i % np.size(prm,1) != 0:
            M[i][i-1] = 1/(h**2)
        if i < len(prm) - 1 & i % np.size(prm,1) != np.size(prm,1) - 1:
            M[i][i+1] = 1/(h**2)
        if i >= np.size(prm,1):
            M[i][i-np.size(prm,1)] = 1/(h**2)
        if i < len(prm) - np.size(prm,1):
            M[i][i+np.size(prm,1)] = 1/(h**2)
    M  = (1/(k0**2)) * M
    eff_eps, guided = sps.linalg.eigs(M, k = numb)
    return eff_eps, guided

# Define the parameters
grid_size     = 120
number_points = 301
h             = grid_size/(number_points - 1)
lam           = 0.78
k0            = 2*np.pi/lam
e_substrate   = 2.25
delta_e       = 1.5e-2
w             = 15.0
xx            = np.linspace(-grid_size/2,grid_size/2,number_points)
yy            = np.linspace(-grid_size/2,grid_size/2,number_points)
XX,YY         = np.meshgrid(xx,yy)
prm           = e_substrate + delta_e * np.exp(-(XX**2+YY**2)/w**2)
numb          = 1

# Compute the eigenvalues and eigenvectors
eff_eps, guided = guided_modes_2D(prm, k0, h, numb)

X, Y = XX, YY
Z = np.real(guided[0].reshape((number_points, number_points)))

# Plot the eigenmode
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(np.min(Z), np.max(Z))
ax.zaxis.set_major_locator(LinearLocator(10))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()