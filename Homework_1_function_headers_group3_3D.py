'''Homework 1, Computational Photonics, SS 2024:  FD mode solver. 1+1=2D
'''
from datetime import timedelta
import time
start_time = time.perf_counter()

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter

# def matrix_gen(n):
#     # Caclulates NxN matrix where N = n**2
#     N = n**2
#     Diagonal_0 = -4*np.ones(N)
#     Diagonal_1 = np.ones(N-1) #adjacent diagonals to center
#     for i in range(N-1): 
#         if (i%n == n-1):Diagonal_1[i] = 0 #corrects side diagonals
#     Diagonal_far = np.ones(N) #side diagnols
#     Matrix = sps.spdiags(Diagonal_0,0,N,N)
#     Matrix += sps.spdiags(np.append(3,Diagonal_1),1,N,N) + sps.spdiags(Diagonal_far,1*n,N,N) #final sparse matrix
#     Matrix += sps.spdiags(Diagonal_1,-1,N,N) + sps.spdiags(Diagonal_far,-1*n,N,N) #final sparse matrix
#     return Matrix #returns sparse matrix 

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
    # M  = np.zeros((np.size(prm), np.size(prm)), dtype = dt)
    # for i  in range(np.size(prm)):
    #     M[i][i] =  -4/(h**2) + (k0**2) * prm.flatten()[i]
    #     if i > 0 & i % np.size(prm,1) != 0:
    #         M[i][i-1] = 1/(h**2)
    #     if i < len(prm) - 1 & i % np.size(prm,1) != np.size(prm,1) - 1:
    #         M[i][i+1] = 1/(h**2)
    #     if i >= np.size(prm,1):
    #         M[i][i-np.size(prm,1)] = 1/(h**2)
    #     if i < len(prm) - np.size(prm,1):
    #         M[i][i+np.size(prm,1)] = 1/(h**2)
    # M  = (1/(k0**2)) * M
    # Set the diagonal elements
    diagonals = np.zeros((5, np.size(prm)))
    diagonals[0] = -4/(h**2) + (k0**2) * prm.flatten()
    for i in range(np.size(prm) - 1):
        if i % len(prm[0]) != len(prm[0]) - 1:
            diagonals[1][i] = 1/(h**2)
        if i % len(prm[0]) != 0:
            diagonals[2][i] = 1/(h**2)
        if i < np.size(prm) - len(prm[0]):
            diagonals[3][i] = 1/(h**2)
        if i >= len(prm[0]):
            diagonals[4][i] = 1/(h**2)
    diag_position = [0, 1, -1, len(prm[0]), -len(prm[0])]
    # Create a sparse 2D array with the diagonal elements
    M = sps.diags(diagonals, diag_position)
    M = (1/(k0**2)) * M
    # Compute the eigenvalues and eigenvectors
    eff_eps, guided = sps.linalg.eigs(M, k = numb, which = 'LR')
    return eff_eps, guided

# Define the parameters
grid_size     = 120
number_points = 201
h             = grid_size/(number_points - 1)
lam           = 0.78
k0            = 2*np.pi/lam
e_substrate   = 2.25
delta_e       = 1.5e-2
w             = 15.0
xx            = np.linspace(-grid_size/2 - h,grid_size/2 + h,number_points + 2)
yy            = np.linspace(-grid_size/2,grid_size/2,number_points)
XX,YY         = np.meshgrid(xx,yy)
prm           = e_substrate + delta_e * np.exp(-(XX**2+YY**2)/w**2)
numb          = 1

# Compute the eigenvalues and eigenvectors
eff_eps, guided = guided_modes_2D(prm, k0, h, numb)
guided = np.transpose(guided)
# print(eff_eps)
# print(guided)

# Calculate the operational time of the program (finding the eigenvalues and eigenvectors)
end_time=time.perf_counter()
print('The operational time of the program is %s seconds' %(end_time-start_time))

mode_ind = 0
print(eff_eps)
# print(guided)

# Plot the eigenmode(3d plot)
X, Y = XX, YY
Z = np.real(guided[mode_ind].reshape((number_points , number_points + 2)))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,6))
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
## Customize the z axis.
ax.set_zlim(np.min(Z), np.max(Z))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.set_title('Guided mode field distribution \n effective permittivity = ' + str(eff_eps[mode_ind]))
ax.set_xlabel('x [µm]', fontsize=10)
ax.set_ylabel('y [µm]', fontsize=10)
ax.set_zlabel('Electric field strength [V/µm]', fontsize=10)
Z_formatter = FormatStrFormatter('%.3f')
ax.zaxis.set_major_formatter(Z_formatter)
## Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Plot the eigenmode(2d plot)
X, Y = XX, YY
Z = np.real(guided[mode_ind].reshape((number_points , number_points + 2)))
fig, ax = plt.subplots(figsize=(6,8))
im = ax.imshow(Z, cmap='coolwarm', extent=[-grid_size/2 - h , grid_size/2 + h, -grid_size/2, grid_size/2])
fig.colorbar(im, ax=ax, label='Electric field strength [V/µm]')
ax.set_title('Guided mode field distribution \n effective permittivity = ' + str(eff_eps[mode_ind]))
ax.set_xlabel('x/µm')
ax.set_ylabel('y/µm')
plt.show()