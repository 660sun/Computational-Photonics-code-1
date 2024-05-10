'''Homework 1, Computational Photonics, SS 2024:  FD mode solver. 1+1=2D
'''
from datetime import timedelta
import time
start_time = time.perf_counter()

import numpy as np
import matplotlib.pyplot as plt

def guided_modes_1DTE(prm, k0, h):
    """Computes the effective permittivity of a TE polarized guided eigenmode.
    All dimensions are in µm.
    Note that modes are filtered to match the requirement that
    their effective permittivity is larger than the substrate (cladding).
    
    Parameters
    ----------
    prm : 1d-array
        Dielectric permittivity in the x-direction
    k0 : float
        Free space wavenumber
    h : float
        Spatial discretization
    
    Returns
    -------
    eff_eps : 1d-array
        Effective permittivity vector of calculated modes
    guided : 2d-array
        Field distributions of the guided eigenmodes
    """
    dt = np.common_type(np.array([prm]))
    M  = np.zeros((len(prm),len(prm)), dtype = dt)
    for i in range(len(prm)):
        M[i][i] =  -2/(h**2) + (k0**2) * prm[i]
        if i > 0:
            M[i][i-1] = 1/(h**2) 
        if i < len(prm) - 1:
            M[i][i+1] = 1/(h**2)
    M  = (1/(k0**2)) * M
    eff_eps, guided = np.linalg.eig(M)
    return eff_eps, guided

# Define the parameters
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

# Compute the eigenvalues and eigenvectors
eff_eps, guided = guided_modes_1DTE(prm, k0, h)

# Filter the eigenvalues and eigenvectors
## Find the indices of the eigenvalues within the given range
indices = np.where((eff_eps >= e_substrate) & (eff_eps <= e_substrate + delta_e))[0]

## Extract the eigenvalues and eigenvectors within the range
selected_eff_eps = eff_eps[indices]
selected_guided = (np.transpose(guided))[indices]
# mode_ind = int(1)
# print('selected_eff_eps: ', selected_eff_eps[mode_ind])
# print('selected_guided: ', selected_guided[mode_ind])

# Calculate the operational time of the program (finding the eigenvalues and eigenvectors)
end_time=time.perf_counter()
print('The operational time of the program is %s seconds' %(end_time-start_time))

mode_ind = int(1)
print('selected_eff_eps: ', selected_eff_eps[mode_ind])
print('selected_guided: ', selected_guided[mode_ind])

# Plot the eigenvalues and eigenvectors
x = xx
y = selected_guided[mode_ind]
plt.plot(x, y)
plt.xlabel('Position [µm]')
plt.ylabel('Electric field strength [V/µm]')
plt.title('Guided mode field distribution \n effective permittivity = ' + str(selected_eff_eps[mode_ind]))
plt.show()