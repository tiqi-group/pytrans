#!/usr/bin/env python3
# from ETH3dTrap import ETH3dTrap as Moments
from pytrans.trap_model.segtrap import ETH3dTrap as Moments

from pytrans.constants import *
from pytrans.global_settings import *
import timeit
import json
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import mpl_toolkits.mplot3d
import matplotlib.animation as anim
import scipy.io as sio
import scipy.signal as ssig
import scipy.interpolate as sintp
import scipy.misc as smis
import scipy.optimize as sopt
import cvxpy as cvy
import os
import pdb
import pickle
import warnings
st = pdb.set_trace

# #### this should absolutely not be here
#
# trap_mom = Moments()  # Global trap moments
# trap_mom.overwriteGlobalVariables()  # makes sure the global variables correspond to the trap variables - only needed if not executed in Moments constructor
#
# ####

# You must copy global_settings.py.example to global_settings.py and
# modify the options locally for your installation.

# Indexing convention:
# Electrodes T1 = 0, T2 = 1, ... T15 = 14; B1 = 15, B2 = 16, ... B15 = 29
# DEATH channels (top-right downwards, leftwards; bottom DEATHs 0->15, top DEATHs 16->31

# Electrode mapping of the trap simulations:
# DCCc(n) corresponds to T(n+1) = Index n
# DCCa(n) corresponds to B(n+1) = Index n+15, i.e. DCCa14 corresponds to B15 = Index 29

# Values represent indices of which electrode each DEATH output
# drives, from 0->31. E.g. dac_channel_transform[5] = 1 tells us that
# DEATH output 5 drives Electrode 1.
dac_channel_transform = np.array([0, 15, 3, 18, 1, 16, 4, 19,   2, 17, 5, 20, -7, 14, 6, 21,
                                  11, 26, 7, 22, 12, 27, 8, 23,  13, 28, 9, 24, -22, 29, 10, 25])
num_elecs = dac_channel_transform.size

# locations of electrode voltages in the waveform files produced by
# the system right now (0 -> 29) (i.e. values represent which DEATH
# output drives each electrode, from 0 -> 29)

# This array is written to geometrically show which electrodes
# are controlled by which DEATH channels.
# Grouping of 4, 3, 1, 3, 4 refers to load, split, exp, split, load.
physical_electrode_transform = np.array([0, 4, 8, 2,  6, 10, 14,  18,  22, 26, 30,  16, 20, 24, 13,
                                         1, 5, 9, 3,  7, 11, 15,  19,  23, 27, 31,  17, 21, 25, 29])

# indices of electrodes to be used for each DAC channel in the waveform file (0 -> 31)
# (i.e. which electrodes does each DEATH channel drive)

# Example: there are 6 DEATH channels, and 4 physical
# electrodes. DEATH channel 1 connects to Electrode 3, DEATH channel 2
# connects to Electrode 4, DEATH channel 3 monitors (produces the same
# output as) Electrode 1, and DEATH channel 4 monitors Electrode
# 2. DEATH channels 5 and 6 connect to Electrodes 1 and 2
# respectively.
#
# For this arrangement, the mapping would be (negative == monitoring;
# zero-indexed)
# dac_channel_transform = [2,3,-0,-1,0,1]
# physical_electrode_transform = [4, 5, 0, 1]

# DEATH channel max voltage outputs

# trap specific Momentsclasses might overwrite these with trap default
# It is encouraged to change the defaults in the trap/MomentsClass or to create a custom Momentsclass (which might inherit everything but the changed settings)
max_elec_voltage = 8.9
max_elec_voltages = np.zeros(30) + max_elec_voltage
max_death_voltages = max_elec_voltages[dac_channel_transform]

min_elec_voltages = -max_elec_voltages
min_death_voltages = -max_death_voltages

# fudge factor to avoid true hardware limit; Ionizer will disallow voltages with a value of max_elec_voltages + max_overhead (or min_elec_voltages - max_overhead). Advised to be nonzero, otherwise the solvers will be overconstrained.
max_overhead = 0.09

# Maximum number of samples the DEATH RAMs can hold
max_death_samples = 16384

# Global default electrode voltage, for solvers etc
# trap specific Momentsclasses might overwrite this with trap default
default_elec_voltage = 5


# Electrode starts and ends in um, ordered from Electrode 0 -> 29
# trap specific Momentsclasses might overwrite this with trap default
electrode_coords = np.array([[-3055, -2055], [-2035, -1535], [-1515, -1015], [-995, -695], [-675, -520], [-500, -345], [-325, -170], [-150, 150], [170, 325], [345, 500], [520, 675], [695, 995], [1015, 1515], [1535, 2035], [2055, 3055], [-3055, -2055], [-2035, -1535], [-1515, -1015], [-995, -695], [-675, -520], [-500, -345], [-325, -170], [-150, 150], [170, 325], [345, 500], [520, 675], [695, 995], [1015, 1515], [1535, 2035], [2055, 3055]])


def roi_potential(potential, z_axis, roi_centre, roi_width):
    assert len(potential) == len(z_axis), "Potential and z axis inputs are of different lengths"
    roi_l = roi_centre - roi_width
    roi_r = roi_centre + roi_width
    roi_idx = (roi_l <= z_axis) & (z_axis <= roi_r)
    pot_roi = np.ravel(potential[roi_idx])
    z_axis_roi = z_axis[roi_idx]
    z_axis_idx = np.arange(z_axis.shape[0])[roi_idx]
    return pot_roi, z_axis_roi, z_axis_idx


def find_coulomb_wells(samples, roi_centre, roi_width, mass=mass_Ca, ions=2, plot_results=False):
    # Similar to find_wells_from_samples, however numerically finds
    # the locations and frequencies of some number of ions when
    # subject to mutual Coulomb repulsion.
    #
    # samples: DEATH voltages in a column vector
    # roi_centre: central point of region of interest (in m)
    # roi_width: +/- from central point (in m); ROI is [roi_centre-roi_width, roi_centre+roi_width]
    #
    # Note: potential within ROI should be a more-or-less even function with
    # the minimum/minima near the middle (e.g. a positive parabola or
    # double-well quartic), otherwise solver may not converge.

    trap_freqs_no_coulomb = []
    trap_freqs = []
    trap_locs_no_coulomb = []
    trap_locs = []
    trap_offsets_no_coulomb = []
    trap_offsets = []

    start_guess = find_wells_from_samples(samples, roi_centre, roi_width)
    num_wells = len(start_guess['freqs'])
    assert 1 <= num_wells, "Too few wells found in ROI: widen ROI or check the potential."
    assert num_wells <= ions, "Too many wells found in ROI: reduce ROI or check the potential."

    for k in range(ions):
        # wrap around/duplicate results for each ion if there are fewer wells than ions
        guess_idx = np.mod(k, num_wells)
        trap_locs_no_coulomb.append(start_guess['locs'][guess_idx])
        trap_freqs_no_coulomb.append(start_guess['freqs'][guess_idx])
        trap_offsets_no_coulomb.append(start_guess['offsets'][guess_idx])

    potential = np.dot(trap_mom.potentials[:, :len(physical_electrode_transform)],
                       samples[physical_electrode_transform])

    pot, z_axis, _ = roi_potential(potential, trap_mom.transport_axis, roi_centre, roi_width)
    # Potential interpolation type; create a callable electrode potential function
    interp_splines = True
    if interp_splines:
        # s_opt = len(z_axis)*1e-12
        s_opt = len(z_axis) * 5e-12
        # s_opt = 0
        pot_fn = sintp.UnivariateSpline(z_axis, pot, s=s_opt, k=5)
    else:
        pot_fn = sintp.interp1d(z_axis, pot, kind='quadratic')

    def pot_coulomb(z0, z1):
        return 2 * electron_charge / (4 * np.pi * epsilon_0) / np.abs(z1 - z0)  # 2x because 2 ions

    def total_pot(z):
        # Calculate total potential (electrodes + ion-ion Coulomb repulsion)
        # z: array of ion positions (NOTE: only implemented for 2 ions for now!)
        # z[1] must be > z[0]
        z0, z1 = z[0], z[1]
        z_min, z_max = z_axis[0], z_axis[-1]
        if z1 <= z0 or z0 <= z_min or z1 >= z_max:
            # should be way higher than any conceivable
            # Coulomb-related potential, to discourage solver
            return 10  # 10 eV should be way above anything normal

        pot_electrodes = pot_fn(z0) + pot_fn(z1)

        # Coulomb repulsion
        return pot_electrodes + pot_coulomb(z0, z1)

    minimize_result = sopt.minimize(total_pot,
                                    [roi_centre, roi_centre + 0.1 * um],
                                    method='Nelder-Mead',
                                    options={'xtol': 1e-12})

    if not minimize_result.success:
        warnings.warn("Coulomb solver did not converge, results may be untrustworthy!")

    # ion positions
    z0, z1 = minimize_result.x[0], minimize_result.x[1]

    # local curvatures at ion positions (using the full potential well)
    v0dd = smis.derivative(lambda z: total_pot((z, z1)), z0, 0.1 * um, n=2, order=5)
    v1dd = smis.derivative(lambda z: total_pot((z0, z)), z1, 0.1 * um, n=2, order=5)

    trap_locs.append(z0)
    trap_locs.append(z1)

    offsets = pot_fn((z0, z1)) + pot_coulomb(z0, z1)
    trap_offsets.append(offsets[0])
    trap_offsets.append(offsets[1])

    def curv2freq(vdd):
        # vdd: double derivative of potential w.r.t. distance (in V/m^2)
        # returns trap freq
        return np.sqrt(electron_charge * vdd / (mass * atomic_mass_unit)) / 2 / np.pi

    trap_freqs.append(curv2freq(v0dd))
    trap_freqs.append(curv2freq(v1dd))

    if plot_results:
        z_hires = np.linspace(z_axis[0], z_axis[-1], 1e3)
        print(z0 / um, z1 / um)
        plt.figure()
        plt.plot(z_axis, pot, 'g')
        plt.plot(z_hires, pot_fn(z_hires), 'b')
        plt.plot([z0, z1], pot_fn([z0, z1]), 'or')
        plt.plot(start_guess['locs'], pot_fn(start_guess['locs']), 'ob')

        # plt.figure()
        # plt.plot(z_hires[4:-4], smis.derivative(pot_fn, z_hires[4:-4], 0.1*um, n=2, order=5), 'r')
        plt.show()

    return {'freqs_no_coulomb': trap_freqs_no_coulomb,
            'freqs': trap_freqs,
            'locs': trap_locs,
            'locs_no_coulomb': trap_locs_no_coulomb,
            'offsets': trap_offsets,
            'offsets_no_coulomb': trap_offsets_no_coulomb}


def calc_potential(samples):
    # Calculate trap potential, ignoring shims, along full trap length
    return np.dot(trap_mom.potentials[:, :len(physical_electrode_transform)],
                  samples[physical_electrode_transform])


def find_wells_from_samples(samples, roi_centre, roi_width):
    # Convenience function to avoid having to generate a WavPotential
    # class for every sample (uses identical code)
    # TODO: extend arguments
    potential = calc_potential(samples)
    return find_wells(potential, trap_mom.transport_axis, mass_Ca, mode='precise',
                      roi_centre=roi_centre, roi_width=roi_width)


def find_wells(potential, z_axis, ion_mass, mode='quick', smoothing_ratio=80, polyfit_ratio=60, freq_threshold=10 * kHz, roi_centre=0 * um, roi_width=2356 * um):
    """For a given potential, return the location of the potential minima, their offsets and curvatures.
    potential: spatial potential vector
    mode: 'quick' or 'precise'.
    smoothing_ratio: fraction of total length of potentials vector to smoothe over (not used atm?!)
    polyfit_ratio: fraction of total length of potentials vector to fit a polynomial to
    freq_threshold: Minimum trapping frequency of the wells to store"""

    assert mode is 'quick' or mode is 'precise', "Input argument 'mode' only supports mode='quick' and mode='precise'"

    # Extract potential within region of interest
    pot, trap_axis_roi, trap_axis_idx = roi_potential(potential, z_axis, roi_centre, roi_width)
    pot_resolution = z_axis[1] - z_axis[0]

    potg2 = np.gradient(np.gradient(pot))
    # Ad-hoc filtering of the waveform potential with a top-hat
    # window 1/80th as big
    # potg2_filt = np.convolve(potg2,
    #                         np.ones(pot.size/smoothing_ratio)/(pot.size*smoothing_ratio),
    #                         mode='same')
    min_indices_candidates, = ssig.argrelextrema(pot, np.less_equal, order=20)  # find relative minima. order=x to suppress detecting most spurious minima

    # If present, remove spurious candidates at boundaries
    start_end_idx = np.array([0, len(pot) - 1], dtype='int')
    min_indices_candidates = np.setdiff1d(min_indices_candidates, start_end_idx)

    # Gather wells
    min_indices = []
    offsets = []
    polys = []
    trap_freqs = []
    trap_locs = []
    for mi in min_indices_candidates:
        if mode is 'quick':
            # numerically evaluate gradient from the raw data (noisy)
            grad = potg2[mi] / pot_resolution**2
            # grad = potg2_filt[mi]/pot_resolution**2
        elif mode is 'precise':
            # Select region of interest (preventing out of bounds errors)
            idx1 = mi - 5 if mi - 5 > 0 else 0
            idx2 = mi + 5 if mi + 5 < pot.shape[0] else pot.shape[0]

            # Fit quadratic to potential
            pfit = np.polyfit(trap_axis_roi[idx1:idx2], pot[idx1:idx2], 2)
            poly = np.poly1d(pfit)
            grad = 2 * poly[2]  # in eV

        # Only keep wells that are confining and above the threshold
        if grad > 0:
            freq = np.sqrt(electron_charge * grad / (ion_mass * atomic_mass_unit)) / 2 / np.pi
            if freq > freq_threshold:
                min_indices.append(trap_axis_idx[mi])
                if mode is 'quick':
                    offsets.append(pot[mi])
                    trap_freqs.append(freq)
                    trap_locs.append(trap_axis_roi[mi])
                elif mode is 'precise':
                    polys.append(poly)
                    offsets.append(-poly[1]**2 / 4 / poly[2] + poly[0])
                    trap_freqs.append(freq)
                    trap_locs.append(-poly[1] / 2 / poly[2])

    return {'min_indices': min_indices, 'offsets': offsets, 'freqs': trap_freqs, 'locs': trap_locs}


def animate_wavpots(wavpots, parallel=True, decimation=10, save_video_path=None):
    # wavpots: must be an iterable of similar WavPotentials
    #
    # parallel: whether to animate the waveforms sequentially or at the same time.
    # If at the same time, their samples matrices must be the same size.
    #
    # decimation: factor reduction in sample number
    #
    # save_video_path: if not None, saves the video to this path.
    Writer = anim.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist="vnegnev"), bitrate=1800)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([-4, 4])
    ax.grid(True)
    ax.set_xlabel('trap location (um)')
    ax.set_ylabel('potential (V)')

    if parallel:
        lines = tuple(ax.plot(k.trap_axis / um, k.potentials[:, 0])[0] for k in wavpots)
        timesteps = wavpots[0].potentials.shape[1]

        def data_gen(i):
            return tuple(wf.potentials[:, i * decimation] for wf in wavpots)

        def animate(i):
            ydata = data_gen(i)
            for line, y in zip(lines, ydata):
                line.set_ydata(y)
    else:  # sequential
        wavpot1 = wavpots[0]
        merged_potentials = np.hstack((wf.potentials for wf in wavpots))
        timesteps = merged_potentials.shape[1]
        line, = ax.plot(wavpot1.trap_axis / um, merged_potentials[:, 0])  # first potential

        # def data_gen(i):
        #     return merged_potentials[:,[i]]

        def animate(i):
            line.set_ydata(merged_potentials[:, [i]])

    im_ani = anim.FuncAnimation(fig, animate,
                                frames=timesteps // decimation,
                                interval=30, blit=False)

    if save_video_path:
        im_ani.save(save_video_path, fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


class WavPotential:
    """ Class for calculating, analyzing and visualizing the potentials resulting
    from the electrodes in both 1d (i.e. along the trap axis), 2d (i.e in the radial plane)
    and 3d (i.e. along the axial and radial directions). Generally to be used for analysing
    and plotting existing waveforms"""

    def __init__(self, waveform, trap_mom, ion_mass=mass_Ca, rf_v=385, rf_freq=115.102, shim_alpha=0, shim_beta=0):

        # Load relevant electrodes and reorder as needed
        # Warning: This is not very robust at the moment! Beware when modifying
        # physical_electrode_transform. Because we probably won't have to change
        # it in a long time, this is implemented sloppily for now.
        mom_trunc = trap_mom.potentials[:, :len(physical_electrode_transform)]
        waveform_samples_trunc = waveform.samples[physical_electrode_transform, :]

        # Assign arguments
        self.waveform_samples = waveform_samples_trunc
        self.potentials = np.dot(mom_trunc, waveform_samples_trunc)  # Potentials along trap axis for all timesteps, [location_idx,timestep_idx]
        self.trap_axis = trap_mom.transport_axis
        self.ion_mass = ion_mass  # (amu)
        self.rf_v = rf_v  # (Volts)
        self.rf_freq = rf_freq  # (MHz) (!!)
        self.shim_alpha = shim_alpha  # (Volts)
        self.shim_beta = shim_beta  # (Volts)

        self.trap_mom = trap_mom

    # Functions for analyzing/plotting the potential along the trap axis

    def plot(self, style='img', ax=None):
        """ Plot the whole waveform in 3D.
        style: either 'img' or 'surf'; controls the plot shown.
        ax: Matplotlib axes """
        trap_axis_spacing = self.trap_axis[1] - self.trap_axis[0]
        # Since plot is showing quadrilaterals
        trap_axis_pts = np.append(self.trap_axis, self.trap_axis[-1] + trap_axis_spacing) \
            - trap_axis_spacing / 2
        trap_axis_pts *= 1e6  # convert from m to um
        px, py = np.meshgrid(np.arange(self.potentials.shape[1] + 1), trap_axis_pts)

        if style == 'img':
            if not ax:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
            pcm = ax.pcolormesh(px, py, self.potentials, cmap='coolwarm')
        elif style == 'surf':
            if not ax:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection='3d')
            px, py = np.meshgrid(np.arange(self.potentials.shape[1]), trap_axis_pts[:-1])  # without the edge points above, since centred now
            pcm = ax.plot_surface(px, py, self.potentials, cmap='coolwarm')
        fig.colorbar(pcm)
        ax.set_xlabel('timestep')
        ax.set_ylabel('trap z axis (um)')
        # ax.colorbar()

    def plot_one_wfm(self, idx, ax=None):
        """ Plot the trapping well along the trap axis at a single time-step.
        idx: time-step
        ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.trap_axis / um, self.potentials[:, idx])
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        return ax

    def plot_voltages(self, timesteps=-1, electrode_idx=range(15), ax=None):
        """ Plot the electrode voltages as a function of timestep.
        timesteps: int or iterable. Int: Number of timesteps, Iterable: desired timesteps. If -1, then plot every timestep.
        electrode_idx: index(es) of the electrode(s) to plot
        ax: Matplotlib axes """
        if type(timesteps) is int:
            # integer, specifying number of points
            if timesteps == -1:
                timesteps = self.potentials.shape[1]
            time_idces = np.linspace(0, self.potentials.shape[1] - 1, timesteps, dtype='int')
        else:
            # iterable
            time_idces = timesteps

        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        ax.grid(True)
        time_xrange = time_idces[-1] - time_idces[0]
        ax.set_xlim([time_idces[0] - 0.3 * time_xrange, time_idces[-1]])
        if type(electrode_idx) is int:
            idx = [electrode_idx]
        for id in electrode_idx:
            ax.plot(time_idces, self.waveform_samples[id, time_idces])
        ax.legend(electrode_idx)

    def plot_electrodes(self, idx=range(15), ax=None):
        """ Plot the potential along the trap axis due to individual electrodes.
        idx: index(es) of the electrode(s) to plot
        ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        ax.grid(True)
        if type(idx) is int:
            idx = [idx]
        for id in idx:
            ax.plot(self.trap_axis / um, self.trap_mom.potentials[:, id])
        ax.legend(idx)

    def plot_range_of_wfms(self, timesteps=-1, ax=None):
        """ Plot the potential along the trap axis at various timesteps.
        timesteps: int or iterable. Int: Number of timesteps, Iterable: desired timesteps. If -1, then plot every timestep.
        ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        if type(timesteps) is int:
            # integer, specifying number of points
            if timesteps == -1:
                timesteps = self.potentials.shape[1]
            idces = np.linspace(0, self.potentials.shape[1] - 1, timesteps, dtype='int')
        else:
            # iterable
            idces = timesteps
        for idx in idces:
            ax.plot(self.trap_axis / um, self.potentials[:, idx])
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        return ax

    def animate_wfm(self, decimation=10, wdp=None):
        """Decimation: how many timesteps to skip for every frame (set to 1 to
        see every timestep)

        wdp: the WavDesiredPotential object used to
        generate the Waveform; will be plotted alongside the actual
        waveform if specified

        """
        Writer = anim.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist="vnegnev"), bitrate=1800)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([-2350, 2350])
        ax.set_ylim([-3, 5])
        ax.grid(True)
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')

        pot_line, = ax.plot(self.trap_axis / um, self.potentials[:, 0])
        lines = [pot_line]
        if wdp:
            des_line, = ax.plot(self.trap_axis[wdp.roi_idx[0]] / um, wdp.potentials[0])
            lines.append(des_line)

        def update(data):
            for l, (dx, dy) in zip(lines, data):
                l.set_data(dx, dy)
            return lines

        if wdp:
            def data_gen():
                for pot, des_roi_idx, des_pot in zip(self.potentials.T[::decimation],
                                                     wdp.roi_idx[::decimation],
                                                     wdp.potentials[::decimation]):
                    yield [(self.trap_axis / um, pot), (self.trap_axis[des_roi_idx] / um, des_pot)]
        else:
            def data_gen():
                for pot in self.potentials.T[::decimation]:
                    yield [(self.trap_axis / um, pot)]

        im_ani = anim.FuncAnimation(fig, update, data_gen, interval=30)  # interval: ms between new frames
        plt.show()

    def find_wells(self, time_idx, mode='quick', smoothing_ratio=80, polyfit_ratio=60, freq_threshold=10 * kHz, roi_centre=0 * um, roi_width=2356 * um):
        """ Wrapper for pytrans.find_wells() """
        return find_wells(self.potentials[:, time_idx], self.trap_axis, self.ion_mass, mode, smoothing_ratio, polyfit_ratio, freq_threshold, roi_centre, roi_width)

    # Functions for analyzing/plotting the potential in 2d (radials), and 3d (axial+radials)

    def add_potentials(self, time_idx, slice_ind=None):
        """ Adds potentials"""

        # Calculate the potential over the full region if no slice is specified
        if slice_ind is None:
            slice_ind = np.arange(trap_mom.pot3d.ntot)

        # Add potential due to RF electrodes
        rf_scaling = self.rf_v**2 / (self.ion_mass * self.rf_freq**2)
        potential = trap_mom.pot3d.potentials['RF_pondpot_1V1MHz1amu'][slice_ind] * rf_scaling

        # Add potential due to DC control electrodes
        for el_idx in range(len(physical_electrode_transform)):
            # Get electrode name
            assert (0 <= el_idx) and (el_idx <= 29), "Electrode index out of bounds"
            if el_idx < 15:
                # Top electrode, thus DCCc (see comment at the top)
                electrode_name = 'DCCc' + str(el_idx)
            else:
                # Bottom electrode, thus DCCa
                electrode_name = 'DCCa' + str(el_idx - 15)

            potential += self.waveform_samples[el_idx, time_idx] * trap_mom.pot3d.potentials[electrode_name][slice_ind]

        # Add potential due to shims
        if self.shim_alpha is not 0 or self.shim_beta is not 0:
            Vsa = self.shim_alpha / 4 + self.shim_beta / 4
            Vsb = -self.shim_alpha / 4 + self.shim_beta / 4
            Vsc = self.shim_alpha / 4 - self.shim_beta / 4
            Vsd = -self.shim_alpha / 4 - self.shim_beta / 4

            potential += Vsa * trap_mom.pot3d.potentials['DCSa'][slice_ind] + \
                Vsb * trap_mom.pot3d.potentials['DCSb'][slice_ind] + \
                Vsc * trap_mom.pot3d.potentials['DCSc'][slice_ind] + \
                Vsd * trap_mom.pot3d.potentials['DCSd'][slice_ind]

        return potential

    def find_radials_2d(self, time_idx):
        """Calculates the axial and radial trap frequencies assuming the axial direction to be along the trap axis, thus
        reducing the calculations required by only having to fit the radials to the potential V(y,z) rather than V(x,y,z).
        Returns all three trap frequencies, trap centre coordinates and axes."""
        # See find_radials_3d for a detailed description.
        # Notation:
        # V(y,z) = ay^2 + bz^2 + c*yz + d*y + e*z + f

        # 1) Find relevant radial plane by finding minimum along trap axis
        roi_c = 0.5 * (trap_mom.pot3d.x[0] + trap_mom.pot3d.x[-1])
        roi_w = 0.5 * (trap_mom.pot3d.x[-1] - trap_mom.pot3d.x[0])
        axial_wells = self.find_wells(time_idx, mode='precise', roi_centre=roi_c, roi_width=roi_w)

        assert len(axial_wells['locs']) > 0, "Found no trapping well in ROI"
        assert len(axial_wells['locs']) < 2, "Found more than one trapping well in ROI"

        r0_x = axial_wells['locs'][0]
        axial_freq = axial_wells['freqs'][0]

        x_idx = (np.abs(trap_mom.pot3d.x - r0_x)).argmin()
        slice_ind = np.arange(trap_mom.pot3d.ntot).reshape(trap_mom.pot3d.nx, trap_mom.pot3d.ny, trap_mom.pot3d.nz, order='F')[x_idx, :, :]  # relevant indices of flattened array

        V = self.add_potentials(time_idx, slice_ind)
        V = V.reshape(trap_mom.pot3d.ny, trap_mom.pot3d.nz)

        # 2) Determine radial modes
        # Linear least squares fit
        p = np.linalg.lstsq(trap_mom.pot3d.fit_coord2d, V.flatten(order='F'))[0]
        a, b, c, d, e, f = p

        # Extract the trapping frequencies and corresponding axes
        A = np.array([[a, c / 2], [c / 2, b]])
        eigenvalues, axes = np.linalg.eig(A)
        radial_freqs = np.sqrt(eigenvalues * 2 * electron_charge / (self.ion_mass * atomic_mass_unit)) / (2 * np.pi)

        # Extract the trapping location by solving for the point where grad V = 0:
        A2 = np.array([[2 * a, c], [c, 2 * b]])
        b2 = -p[3:5]  # -[d; e; f]
        r0_yz = np.linalg.lstsq(A2, b2)[0]  # Solve A2*r0 = b

        # Overall offset
        offset = f

        # Combine axial + radial information
        omegas = np.zeros(3)
        omegas[0] = axial_freq
        omegas[1:3] = radial_freqs
        if any(w < 0 for w in omegas):
            warnings.warn('Potential is anti-confining.')

        axes = np.zeros((3, 3))
        axes[0, 0] = 1  # put axial eigenvector first
        if radial_axes[1, 0] < 0:  # align both radial eigenvectors in the upper half of the yz plane
            radial_axes[:, 0] = -radial_axes[:, 0]
        if radial_axes[1, 1] < 0:
            radial_axes[:, 1] = -radial_axes[:, 1]
        axes[1:, 1:] = radial_axes
        r0 = np.zeros(3)
        r0[0] = r0_x
        r0[1:3] = r0_yz

        return omegas, axes, r0, offset, V

    def find_radials_3d(self, time_idx):
        """Calculate the three trapping frequencies and orthonormal directions given by the potential V(x,y,z)"""

        # Outline: Assume that the potential V(x,y,z) is quadratic in all directions.
        # If the axes of the coordinate system coincide with the principal axes of
        # the potential and it is centered at the origin, we have
        # V(x,y,z) = a0*x^2 + b0*y^2 + c0*z^2
        # Upon allowing a rotation of the axes, and a translation of the origin, we have
        # (1) V(r) = (r-r0)'*A*(r-r0)
        # where r0 corresponds to the translation and A is a symmetric positive definite matrix.
        # Rewriting (1) in terms of (x,y,z), we get
        # (2) V(x,y,z)  = a*x^2 + b*y^2 + c*z^2 + d*xy + e*xz + f*yz + g*x + h*y + i*z + j
        # Here, we do linear least squares fit to get p = (a,b,...,i,j) and then extract
        # the various properties of the potential from p.

        # 1) Find relevant region of 3d potential by finding minimum along trap axis
        roi_c = 0.5 * (trap_mom.pot3d.x[0] + trap_mom.pot3d.x[-1])
        roi_w = 0.5 * (trap_mom.pot3d.x[-1] - trap_mom.pot3d.x[0])
        axial_wells = self.find_wells(time_idx, mode='precise', roi_centre=roi_c, roi_width=roi_w)

        assert len(axial_wells['locs']) > 0, "Found no trapping well in ROI"
        assert len(axial_wells['locs']) < 2, "Found more than one trapping well in ROI"

        x_idx = (np.abs(trap_mom.pot3d.x - axial_wells['locs'][0])).argmin()
        pot_slice_ind = np.arange(trap_mom.pot3d.ntot).reshape(trap_mom.pot3d.nx, trap_mom.pot3d.ny, trap_mom.pot3d.nz, order='F')[x_idx - 5:x_idx + 5 + 1, :, :].flatten()  # +1 due to slice indexing
        V = self.add_potentials(time_idx, slice_ind=pot_slice_ind)

        # 2) Find axial & radial modes
        # Linear least squares fit
        p = np.linalg.lstsq(trap_mom.pot3d.fit_coord3d[pot_slice_ind, :], V)[0]
        a, b, c, d, e, f, g, h, i, j = p

        # Extract the trapping frequencies and corresponding axes:
        # Idea: Expanding (1), we get V(r) = (r-r0)'*A*(r-r0) = r'*A*r + ...
        # The term r'*A*r corresponds to the terms
        # a*x^2 + b*y^2 + c*z^2 + d*xy + e*xz + f*yz from (2)
        # We can thus read off A from (a, ... ,f)
        # Calculating the eigenvalues and eigenvectors of A then gives us the trap
        # strengths and their associated axes.
        A = np.array([[a, d / 2, e / 2], [d / 2, b, f / 2], [e / 2, f / 2, c]])
        eigenvalues, axes = np.linalg.eig(A)  # each column of axes corresponds to one eigenvector
        freqs = np.sqrt(eigenvalues * 2 * electron_charge / (self.ion_mass * atomic_mass_unit)) / (2 * np.pi)  # Freq in Hz

        if any(w < 0 for w in freqs):
            warnings.warn('Potential is anti-confining.')

        # Extract the trapping location by solving for the point where grad V = 0:
        # dV/dx = 2*a*x + d*y + e*z + g = 0
        # dV/dy = d*x + 2*b*y + f*z + h = 0
        # dV/dz = e*x + f*y + 2*c*z + i = 0
        # A2 = [2a d e; d 2b f; e f 2c], b = -[g; h; i]
        # A2*[x;y;z] = b2
        A2 = np.array([[2 * a, d, e], [d, 2 * b, f], [e, f, 2 * c]])
        b2 = -p[6:9]  # h i j
        r0 = np.linalg.lstsq(A2, b2)[0]  # Solve Ax = b

        # Overall offset
        offset = p[-1]  # j

        # Sort the results such that freqs[0] and axes[:,0] correspond to the axial mode
        axial_mode_idx = np.abs(axes[0, :]).argmax()  # idx of eigenvector with largest component along axial direction x
        if axial_mode_idx != 0:
            if axial_mode_idx == 1:
                permutation = np.array([axial_mode_idx, 0, 2], dtype='int')
            elif axial_mode_idx == 2:
                permutation = np.array([axial_mode_idx, 0, 1], dtype='int')
            # Put axial first
            freqs = freqs[permutation]
            axes = axes[:, permutation]

        # align both radial eigenvectors in the upper half of the yz plane
        if axes[2, 1] < 0:
            axes[:, 1] = -axes[:, 1]
        if axes[2, 2] < 0:
            axes[:, 2] = -axes[:, 2]

        return freqs, axes, r0, offset, V

    def plot_radials(self, time_idx, ax=None, mode='3d', ax_title=None):
        """ Plots the potential in the radial plane together with the radial directions,
        well centre position, and all frequencies."""
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        if mode == '3d':
            freqs, axes, r0, offset, V_3d = self.find_radials_3d(time_idx)
            # Pick required 2d slice from 3d potential:
            V = V_3d.reshape((11, trap_mom.pot3d.ny, trap_mom.pot3d.nz))[5, :, :]
        elif mode == '2d':
            freqs, axes, r0, offset, V = self.find_radials_2d(time_idx)
        else:
            assert False, "Input argument 'mode' only supports mode='2d' or mode='3d'."

        # Align the axes correctly with respect to the imshow function
        V = V.T
        V = np.flipud(V)

        # Plot

        # Scale plot to simulation extent (required due to imshow quirks)
        if trap_mom.pot3d.ntot == 5904:  # trap_exp.pickle, +-100um axially, +-11um radially
            extent = 12
            scalefactor = 7  # length of arrows
        else:  # trap.pickle, +-1000um axially, +-4um radially
            extent = 4.5
            scalefactor = 3  # length of arrows

        res = ax.imshow(V, extent=[-extent, extent, -extent, extent], interpolation='none', cmap='viridis')
        cbar = plt.colorbar(res, fraction=0.046, pad=0.04)  # Numbers ensure cbar has same size as plot
        ax.plot(r0[1] / um, r0[2] / um, 'r.', markersize=10)
        soa = ([r0[1] / um, r0[2] / um, axes[1, 1], axes[2, 1]], [r0[1] / um, r0[2] / um, axes[1, 2], axes[2, 2]])
        X0, Y0, XV, YV = zip(*soa)  # pair up the above
        ax.quiver(X0, Y0, XV, YV, scale_units='xy', scale=1 / scalefactor, color='white')

        # Annotate plot with trap freqs. and origin of well
        ax.text(r0[1] / um + extent / 4, r0[2] / um, 'Ax: ' + '{:.2f}'.format(freqs[0] / MHz) + ' MHz', color='white')  # Axial freq
        ax.text(r0[1] / um + scalefactor * axes[1, 1], r0[2] / um + scalefactor * axes[2, 1], '{:.2f}'.format(freqs[1] / MHz) + ' MHz', color='white')  # Radial 1
        ax.text(r0[1] / um + scalefactor * axes[1, 2], r0[2] / um + scalefactor * axes[2, 2], '{:.2f}'.format(freqs[2] / MHz) + ' MHz', color='white')  # Radial 2
        ax.text(-extent / 4, -extent / 2, 'x0 = ' + '{:.2f}'.format(r0[0] / um) + ' um\n' + 'y0 = ' + '{:.2f}'.format(r0[1] / um) + ' um\n' + 'z0 = ' + '{:.2f}'.format(r0[2] / um) + ' um', color='white')  # Centre locations

        # Format plot
        ax.set_xlabel('y (um)')
        ax.set_ylabel('z (um)')
        plt.xticks(trap_mom.pot3d.y / um)
        plt.yticks(trap_mom.pot3d.z / um)
        ax.set_xlim([-extent, extent])
        ax.set_ylim([-extent, extent])
        cbar.set_label('Potential (V)')
        if not ax_title:
            ax_title = mode + ' analysis of the radials'
        ax.set_title(ax_title)


if __name__ == "__main__":
    # Debugging stuff -- write unit tests from the below at some point
    # wfs.write("waveform_files/test_splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")

    radial_tests = True
    if radial_tests:
        # Some tests showcasing the analysis of the radials
        # Generates a dummy static waveform and then analyzes it.

        wf_path = os.path.join("waveform_files", "radial_tests.dwc.json")

        # If file exists already, just load it to save time
        try:
            raise FileNotFoundError  # uncomment to always regenerate file for debugging
            wfs_load = WaveformSet(waveform_file=wf_path)
            print("Loaded waveform ", wf_path)
        except FileNotFoundError:
            print("Generating waveform ", wf_path)

            local_weights = {'r0': 1e-6,
                             'r0_u_weights': np.ones(30) * 1e-4,
                             'r0_u_ss': np.ones(30) * default_elec_voltage,
                             'r1': 1e-6, 'r2': 1e-7}

            local_potential_params = {'energy_threshold': 10 * meV}

            def static_waveform(pos, freq, offs, wfm_desc):
                wdw = WavDesiredWells([pos * um], [freq * MHz], [offs * meV],
                                      solver_weights=local_weights,
                                      desired_potential_params=local_potential_params,
                                      desc=wfm_desc + ", {:.3f} MHz, {:.1f} meV".format(freq, offs))
                wf = Waveform(wdw)
                return wf

            def transport_waveform(pos, freq, offs, timesteps, wfm_desc, linspace_fn=np.linspace):
                wdw = WavDesiredWells(
                    [linspace_fn(pos[0], pos[1], timesteps) * um],
                    [linspace_fn(freq[0], freq[1], timesteps) * MHz],
                    [linspace_fn(offs[0], offs[1], timesteps) * meV],

                    solver_weights=local_weights,
                    desired_potential_params=local_potential_params,

                    desc=wfm_desc + ", {:.3f}->{:.3f} MHz, {:.1f}->{:.1f} meV".format(freq[0], freq[1], offs[0],
                                                                                      offs[1])
                )
                return Waveform(wdw)

            wf_exp_static_13 = static_waveform(50, 1.8, 500, "static")
            #wf_exp_shallow_16 = transport_waveform([-500, 0], [1.8, 1.8], [1500, 1500], 51, "shallow")
            #wf_exp_shallow_16 = transport_waveform([-500, 0], [1.8, 1.8], [1500, 1500], 501, "shallow")
            wf_list = [wf_exp_static_13]
            wfs_load = WaveformSet(wf_list)
            wfs_load.write(wf_path)

        # Analyze waveform
        WavPot = WavPotential(wfs_load.get_waveform(0), shim_beta=0, shim_alpha=0)

        WavPot.plot_radials(0, mode='2d')
        WavPot.plot_radials(0, mode='3d')

    if False:
        wfs = WaveformSet(waveform_file="waveform_files/splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")
        wfs.write("waveform_files/test2_splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")

    if False:
        # Generate loading waveform
        n_load = 1000
        wdp = WavDesiredWells(
            np.linspace(-1870, 0, n_load) * um,
            np.linspace(1.1, 1.3, n_load) * MHz,
            np.linspace(600, 1000, n_load) * meV,
            desc="Load -> exp test")
        wf1 = Waveform(wdp)
        pot_test = calculate_potentials(trap_mom, wf1)
    #    print(pot_test.find_wells(0))
    #    pot_test.plot()
    #    plt.show()

        wfs = WaveformSet([wf1])
        wfs.write("waveform_files/loading_py_2016_05_23_v01.dwc.json")

        pot_test.plot_one_wfm(0)
        pot_test.plot_one_wfm(-1)
        plt.show()

    if False:
        # Plot the above-generated waveform
        wfs = WaveformSet(waveform_file="waveform_files/loading_py_2016_05_23_v01.dwc.json")
        pot_test = calculate_potentials(trap_mom, wfs.get_waveform(0))
        pot_test.plot_one_wfm(0)
        pot_test.plot_one_wfm(-1)
        plt.show()
