#################################
### Project - 3 
### Name: Divyesh Rathod
### ASU ID: 1225916954
##################################


# Import libraries
import numpy as np 
import pandas as pd 
from scipy.optimize import fsolve
import matplotlib.pyplot as plt 
from scipy import optimize
import warnings
warnings.simplefilter('ignore')


# Constants 

Q = 1.6021766208e-19    # Charge 
KB = 1.380648e-23       # Boltzmann constant
I_SOURCE = 1e-9         # Source Current
N = 1.7                 # Ideality
RES = 11e3              # Resistor
TEMP = 350              # Temperature 
VDD_STEP = 0.1          # Guess diode voltage


### Problem 1 ###

# Calculate Diode Voltage
def solve_diode_v(vol_di, ap_v, res, n, temp, cur_sc):
    cur_di = compute_diode_current(vol_di, n, temp, cur_sc)
    err = (vol_di - ap_v) / res + cur_di
    return err

# Calculate Diode Current
def compute_diode_current(Vd, n, T, i):
    return i * (np.exp(Vd / (n * KB * T / Q)) - 1)

# Creating a ranging of the source voltage(0,1 to 2.5 with step of 0.1 )
source_v = np.arange(0.1, 2.6, 0.1)

# Initialize list to store the voltage and current values
V_diode = []
diode_cur_I = []

# Diode voltages using fsolve function 
for V in source_v:
    root = fsolve(solve_diode_v, VDD_STEP, args=(V, RES, N, TEMP, I_SOURCE), xtol=1e-12)
    VDD_STEP = root[0]
    V_diode.append(root[0])

# Diode currents
for i in V_diode:
    I = compute_diode_current(i, N, TEMP, I_SOURCE)
    diode_cur_I.append(I)

# Setup the parameters for the plot
figure, axis1 = plt.subplots(figsize=(15, 7))

# Label for y-axis and x-axis 
axis1.set_title("Problem 1 plot")
axis1.set_ylabel("Diode Current (log scale)")
axis1.set_xlabel("Voltage in volts")

# To plot between source voltage and log of diode current
axis1.plot(source_v, np.log10(diode_cur_I), label='Source Voltage Vs Diode Current')

# To plot between log of diode current and voltage of diode
axis1.plot(V_diode, np.log10(diode_cur_I), label='Diode Voltage Vs Diode Current')

# To show plot 
axis1.legend()
plt.show()


##############################################################################

### Problem 2 ###

r_val = 10000               # Initial resistance (Ohms)
ide_val = 1.5               # Initial ideality factor (n)
phi_val = 0.8               # Initial barrier height (phi) in eV
P2_AREA = 1e-8              # Diode area (m^2)
P2_TEMP = 375               # Diode temperature (K)
TOLERANCE = 1e-10           # Tolerance for convergence
NITER = 300                 # Max iterations


# Load data from DiodeIV.txt file
data = np.loadtxt('DiodeIV.txt', dtype=np.float64)

# Source Voltage and Diode Current
source_v = data[:, 0]
meas_diode_i = data[:, 1]


def opt_r(r_value, ide_value, phi_value, area, temp, src_v, meas_i):
    est_v = np.zeros_like(src_v)             # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)           # an array to hold the diode currents
    prev_v = VDD_STEP                     # an initial guess for the voltage
    
    # need to compute the reverse bias saturation current for this phi
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )
    
    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,(src_v[index],r_value,ide_value,temp,is_value),xtol=1e-12)[0]
        est_v[index] = prev_v                # store for error analysis
    
    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    return meas_i - diode_i


def opt_phi(phi_val, ide_val, r_val, area, temp, src_volt, i_cur_meas): 
    v_est = np.zeros_like(src_volt)           # an array to hold the diode voltages
    i_cur_diode = np.zeros_like(src_volt)     # an array to hold the diode currents
    v_prev = VDD_STEP                      # an initial guess for the voltage
 
    # need to compute the reverse bias saturation current for this phi
    is_value = area * temp * temp * np.exp(-phi_val * Q / (KB * temp))
    
    for idx in range(len(src_volt)):
        v_prev = optimize.fsolve(solve_diode_v, v_prev, (src_volt[idx], r_val,ide_val, temp, is_value), xtol=1e-12)[0]
        v_est[idx] = v_prev 
 
    # compute the diode current
    i_cur_diode = compute_diode_current(v_est, ide_val, temp, is_value) 
    return_val = (i_cur_meas - i_cur_diode) / (i_cur_meas + i_cur_diode + 1e-15)
    return return_val


def opt_n(ide_val, r_val, phi_val, area, temp, src_volt, i_cur_meas):
    v_est = np.zeros_like(src_volt) # an array to hold the diode voltages
    i_cur_diode = np.zeros_like(src_volt) # an array to hold the diode currents
    v_prev = VDD_STEP # an initial guess for the voltage
 
    # need to compute the reverse bias saturation current for this phi
    is_value = area * temp * temp * np.exp(-phi_val * Q / (KB * temp))
 
    for idx in range(len(src_volt)):
        v_prev = optimize.fsolve(solve_diode_v, v_prev, (src_volt[idx], r_val,ide_val, temp, is_value), xtol=1e-12)[0]
        v_est[idx] = v_prev 
 
 # compute the diode current
    i_cur_diode = compute_diode_current(v_est, ide_val, temp, is_value) 
    return_val = (i_cur_meas - i_cur_diode) / (i_cur_meas + i_cur_diode + 1e-15)
    return return_val


# Initialize iteration and error values
iteration_cnt = 1                              
error = 100                        

while error > TOLERANCE and iteration_cnt < NITER:
    r_val_opt = optimize.leastsq(opt_r, r_val, args=(ide_val, phi_val, P2_AREA, P2_TEMP, source_v, meas_diode_i))
    r_val = r_val_opt[0][0]

    phi_val_opt = optimize.leastsq(opt_phi, phi_val, args=(ide_val, r_val, P2_AREA, P2_TEMP, source_v, meas_diode_i))
    phi_val = phi_val_opt[0][0]

    ide_val_opt = optimize.leastsq(opt_n, ide_val, args=(r_val, phi_val, P2_AREA, P2_TEMP, source_v, meas_diode_i))
    ide_val = ide_val_opt[0][0]
    
    res = opt_phi(phi_val, ide_val, r_val, P2_AREA, P2_TEMP, source_v, meas_diode_i)
    error = np.sum(np.abs(res) / len(res))
    
    print("\nIter: {}; R: {:.2f}; Phi: {:.4f}; Ideality: {:.4f}; Error: {:.3e}".format(iteration_cnt, r_val, phi_val, ide_val, error))
    iteration_cnt += 1


print("\nResistor value (r): {:.2f} Ohms".format(r_val))
print("Barrier height value (phi): {:.2f}".format(phi_val))
print("Final ideality value (n): {:.2f}".format(ide_val))

diode_current = []

for i in range(len(source_v)):
    V = source_v[i]
    curr = P2_AREA * P2_TEMP * P2_TEMP * np.exp(-phi_val * Q / (KB * P2_TEMP))
    root = fsolve(solve_diode_v, VDD_STEP, args = (V, r_val, ide_val, P2_TEMP,curr), xtol=1e-12)
    VDD_STEP = root[0]
    iteration_cnt = compute_diode_current(VDD_STEP, ide_val, P2_TEMP, curr)
    diode_current.append(iteration_cnt)

# Setup the parameters for the plot
figure, ax2 = plt.subplots(figsize=(15, 7))

# Label for y-axis and x-axis 
ax2.set_xlabel("Source voltage in volts")
ax2.set_ylabel("Diode current in log scale")
ax2.set_title("Problem 2 plot")

# To plot between source voltage and log of diode current
ax2.plot(source_v, np.log10(meas_diode_i), label='Source Voltage Vs Measured Diode Current', marker='o')

# To plot between log of diode current and voltage of diode
ax2.plot(source_v, np.log10(diode_current), label='Source Voltage Vs Predicted Diode Current', marker='x', markersize='14')

# To show plot 
ax2.legend()
plt.show()