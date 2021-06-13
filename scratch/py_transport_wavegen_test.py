# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:39:30 2016

First prototype showing how to generate waveforms for transport of single ions.

Note: This seems to work for smallish problems (i.e. < 100 timesteps), but
is very slow or even crashes for larger problems (1000 timesteps). Will have
to investigate this further.

@author: Robin Oswald
"""

# %%
from pytrans import *

# Load moments file
mom = Moments()
num_electrodes = 30

# Specify properties of desired potential well
Ca_amu = 39.962591  # (amu)
frequency = 1.8 * MHz  # (MHz)
offset = 1500 * meV  # (meV)


# Generate desired potential well
def DesiredPotential(position, frequency, offset):
    a = (2 * np.pi * frequency)**2 * (Ca_amu * atomic_mass_unit) / (2 * electron_charge)
    v_desired = a * (mom.transport_axis - position)**2 + offset  # v = a*(z-z0)^2 + b, harmonic well
    EnergyThreshold = 200 * meV
    InsideROIIndices = (v_desired < offset + EnergyThreshold).nonzero()[0]
    return v_desired[InsideROIIndices], InsideROIIndices


# Specify hardware constraints
MaxElectrodeVoltage = 10 * np.ones(num_electrodes)
MinElectrodeVoltage = - MaxElectrodeVoltage
MaxSlewrate = 5 / us  # (Volts/s)

# Cost function parameters
r0 = 1e-4  # punishes deviations from r0_u_ss. Can be used to guide
r1 = 1e-3  # punishes the first derivative of u, thus limiting the slew rate
r2 = 1e-4  # punishes the second derivative of u, thus enforcing smoothness

r0_u_ss = np.ones(num_electrodes)  # default voltage for the electrodes. any deviations from this will be punished, weighted by r0 and r0_u_weights
r0_u_weights = np.ones(num_electrodes)  # use this to put different weights on outer electrodes


# Transport specifications
Ts = 5000 * ns
TransportDuration = 100 * us
N = np.ceil(TransportDuration / Ts + 1).astype(int)  # Number of timesteps
position = np.linspace(-500 * um, 500 * um, N)


#%% Set up and solve optimization problem ###

u = cvy.Variable(num_electrodes, N)
states = []  # empty

for kk in range(N):
    # Cost term capturing how accurately we generate the desired potential well
    v_desired, InsideROIIndices = DesiredPotential(position[kk], frequency, offset)

    cost = cvy.sum_squares(mom.potentials[InsideROIIndices, :] * u[:, kk] - v_desired)
    cost += r0 * cvy.sum_squares(r0_u_weights * (u[:, kk] - r0_u_ss))

    # Absolute voltage constraints
    constr = [MinElectrodeVoltage <= u[:, kk], u[:, kk] <= MaxElectrodeVoltage]

    assert (N < 2) or (N > 3), "Cannot have this number of timesteps, due to finite-diff approximations"

    if N > 3:  # time-dependent constraints require at least 4 samples
        # Approximate costs on first and second derivative of u with finite differences
        # Here, we use 2nd order approximations. For a table with coefficients see
        # https://en.wikipedia.org/wiki/Finite_difference_coefficient
        if (kk != 0 and kk != N - 1):
            # Middle: Use central finite difference approximation of derivatives
            cost += r1 * cvy.sum_squares(0.5 * (u[:, kk + 1] - u[:, kk - 1]))
            cost += r2 * cvy.sum_squares(u[:, kk + 1] - 2 * u[:, kk] + u[:, kk - 1])
        elif kk == 0:
            # Start: Use forward finite difference approximation of derivatives
            cost += r1 * cvy.sum_squares(-0.5 * u[:, kk + 2] + 2 * u[:, kk + 1] - 1.5 * u[:, kk])
            cost += r2 * cvy.sum_squares(-u[:, kk + 3] + 4 * u[:, kk + 2] - 5 * u[:, kk + 1] + 2 * u[:, kk])
        elif kk == N - 1:
            # End: Use backward finite difference approximation of derivatives
            cost += r1 * cvy.sum_squares(1.5 * u[:, kk] - 2 * u[:, kk - 1] + 0.5 * u[:, kk - 2])
            cost += r2 * cvy.sum_squares(2 * u[:, kk] - 5 * u[:, kk - 1] + 4 * u[:, kk - 2] - u[:, kk - 3])

        # Slew rate constraints
        if (kk != N - 1):
            constr += [-MaxSlewrate * Ts <= u[:, kk + 1] - u[:, kk], u[:, kk + 1] - u[:, kk] <= MaxSlewrate * Ts]

    states.append(cvy.Problem(cvy.Minimize(cost), constr))

prob = sum(states)
prob.solve(solver='ECOS', verbose=False)  # ECOS is faster than CVXOPT, but can crash for larger problems

#%% Visualize results ###

# Plot desired and generated potential wells for a few timesteps
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('trap axis z (um)')

for kk in range(0, N, np.ceil(N / 5).astype(int)):
    v_desired, InsideROIIndices = DesiredPotential(position[kk], frequency, offset)
    ax.plot(mom.transport_axis[InsideROIIndices] * 1e6, v_desired, label="desired", color='black')
    ax.plot(mom.transport_axis[InsideROIIndices] * 1e6, mom.potentials[InsideROIIndices, :] * u.value[:, kk], label="actual")

plt.grid(True)
plt.show()

# Plot waveforms
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Timestep ()')

for kk in range(30):
    ax.plot(range(N), u.value[kk, :].transpose())

plt.grid(True)
plt.show()
