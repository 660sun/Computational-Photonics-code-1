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
    # Construct the matrix M for the eigenvalue problem
    M  = np.zeros((len(prm),len(prm)), dtype = dt)
    for i in range(len(prm)):
        M[i][i] =  -2/(h**2) + (k0**2) * prm[i]
        if i > 0:
            M[i][i-1] = 1/(h**2) 
        if i < len(prm) - 1:
            M[i][i+1] = 1/(h**2)
    M  = (1/(k0**2)) * M
    # Compute the eigenvalues and eigenvectors
    eff_eps, guided = np.linalg.eig(M)
    return eff_eps, guided

# Define the basic parameters and the grids
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

# Calculate the operational time of the program (finding the eigenvalues and eigenvectors)
end_time=time.perf_counter()
print('The operational time of the program is %s seconds' %(end_time-start_time))

print('There are ', len(selected_eff_eps), ' modes within the given range.')
mode_ind = int(input('Please input the order of the mode you want to plot:'))
if mode_ind >= 0 & mode_ind < len(selected_eff_eps):
    print('selected_eff_eps: ', selected_eff_eps[mode_ind])
    print('selected_guided: \n', selected_guided[mode_ind])
if mode_ind >= len(selected_eff_eps) | mode_ind < 0:
    print('The input number is out of range. Please input a positive number less than ', len(selected_eff_eps))

# Plot the field distribution of the selected mode and the permittivity
x = xx
y1 = selected_guided[mode_ind]
y2 = prm

fig, ax1 = plt.subplots(figsize=(8, 3))

color = 'tab:blue'
ax1.set_xlabel('Position [µm]')
ax1.set_ylabel('Electric field strength [V/µm]', color=color)
ax1.plot(x, y1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Dielectric permittivity', color=color)
ax2.plot(x, y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Guided mode field distribution \n effective permittivity = ' + str(selected_eff_eps[mode_ind]))
plt.show()

# Convergence and time analysis
n_count = []
time_operation = []
eff_eps_calculated = []

# Define params variable
for i in range(40):
    start_time = time.time()
    number_points = 101 + 5*i
    h             = grid_size/(number_points - 1)
    xx            = np.linspace(-grid_size/2, grid_size/2 ,number_points)
    prm           = e_substrate + delta_e * np.exp(-(xx/w)**2)
    # Compute the eigenvalues and eigenvectors
    eff_eps = guided_modes_1DTE(prm, k0, h)[0]
    max_eff_eps = np.max(eff_eps)
    eff_eps_calculated.append(max_eff_eps)
    n_count.append(number_points)
    time_operation.append(time.time() - start_time)

# Plot convergence of the effective permittivity with increasing number of points
plt.figure(figsize=(5,5)) 
plt.plot(n_count, eff_eps_calculated)
plt.xlabel("Number of points used for calculation")
plt.ylabel("Epsilon")
plt.title("Epsilon as a function of N")
plt.show()

# Plot the operational time with increasing number of points
plt.figure(figsize=(5,5)) 
plt.plot(n_count, time_operation)
plt.xlabel("Number of points used for calculation")
plt.ylabel("Time used for calculation in seconds")
plt.title("Operation time as a function of N")
plt.show()