# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:47:18 2016

Prototype script demonstrating how to generate static waveforms from python using
CVXPY to specify the optimization problem and then call external solvers (e.g
ECOS, CVXOPT, Gurobi, etc.) to find the solution, giving us the electrode voltages.

@author: Robin Oswald
"""

#%% Setup
from test import * # import pytrans functionality
import cvxpy as cvy

# Convenient definitions
um = 1e-6;
MHz = 1e6;
kHz = 1e3;
meV = 1e-3;

# Load moments file (contains e.g. the electrode potentials)
mom = Moments()

# Specify properties of desired potential well
Ca_amu = 40 # [amu]
position = 0*um # [um]
frequency = 1.8*MHz # [MHz]
offset = 1500*meV # [meV]

# Generate desired potential well
def DesiredPotential(position,frequency,offset):
    a = (2*np.pi*frequency)**2*(Ca_amu*atomic_mass_unit)/(2*electron_charge)
    v_desired = a*(mom.transport_axis-position)**2 + offset # v = a*(z-z0)^2 + b, harmonic well
    EnergyThreshold = 200*meV
    InsideROIIndices = (v_desired < offset + EnergyThreshold).nonzero()[0]
    return v_desired[InsideROIIndices],InsideROIIndices

v_desired, InsideROIIndices = DesiredPotential(position, frequency, offset)


#%% Set up and solve optimization problem ###

# Hardware constraints
MaxElectrodeVoltage = 10*np.ones(30)
MinElectrodeVoltage = - MaxElectrodeVoltage

# Set up the problem with cvxpy
u = cvy.Variable(30,1)
cost = cvy.sum_squares(mom.potentials[InsideROIIndices,:]*u-v_desired)
cost += cvy.sum_squares(.005*u)
constraints = [MinElectrodeVoltage <= u, u <= MaxElectrodeVoltage]

for kk in range(15):
    # symmetry constraints for electrode pairs
    constraints += [u[kk] == u[kk+15]]

objective = cvy.Minimize(cost)

prob = cvy.Problem(objective, constraints)
prob.solve()


#%% Visualize results ###

# Plot desired and generated potential
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('trap axis z (um)')
ax.plot(mom.transport_axis[InsideROIIndices]*1e6, v_desired,label="desired",color='black')
ax.plot(mom.transport_axis[InsideROIIndices]*1e6, mom.potentials[InsideROIIndices,:]*u.value,label="actual")

plt.grid(True)
ax.legend()
plt.show()


# Plot trap electrode potentials
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('trap axis z (um)')
ax.set_ylabel('potential (V)')
#C = np.array
for kk in range(30):
    ax.plot(mom.transport_axis*1e6, mom.electrode_moments[kk][:,0]) # electrode potentials along transpor taxis
    
    maxind = np.argmax(mom.electrode_moments[kk][:,0])
    ax.plot(np.array([mom.transport_axis[maxind], mom.transport_axis[maxind]])*1e6,np.array([0, u.value[kk]])/40,marker='1',ms=20) # strength of the various electrodes
    
    
ax.axis([-2500, 2500, 0 , 0.25])
plt.grid(True)
plt.show()
