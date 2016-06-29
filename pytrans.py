#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mpl_toolkits.mplot3d
import matplotlib.animation as anim
import scipy.io as sio
import scipy.signal as ssig
import scipy.stats as sstat
import cvxpy as cvy
import os
import pdb
import pickle
import warnings
st = pdb.set_trace

# You must copy global_settings.py.example to global_settings.py and
# modify the options locally for your installation.
from global_settings import *

# Unit definitions, all in SI
electron_charge = 1.60217662e-19 # coulombs
atomic_mass_unit = 1.66053904e-27 # kg
mass_Ca = 39.962591
epsilon_0 = 8.854187817e-12 # farad/m
um = 1e-6
us = 1e-6
ns = 1e-9
MHz = 1e6
kHz = 1e3
meV = 1e-3

# Indexing convention:
# Electrodes T1 = 0, T2 = 1, ... T15 = 14; B1 = 15, B2 = 16, ... B15 = 29
# DEATH channels (top-right downwards, leftwards; bottom DEATHs 0->15, top DEATHs 16->31

# Electrode mapping of the trap simulations: 
# DCCc(n) corresponds to T(n+1) = Index n
# DCCa(n) corresponds to B(n+1) = Index n+15, i.e. DCCa14 corresponds to B15 = Index 29
                  
# indices of which electrode each DEATH output drives, from 0->31
dac_channel_transform = np.array([0, 15,3,18, 1,16,4,19,   2,17,5,20,-7,14, 6,21,
                                  11,26,7,22,12,27,8,23,  13,28,9,24,-22,29,10,25])
num_elecs = dac_channel_transform.size

# locations of electrode voltages in the waveform files produced by
# the system right now (0 -> 29) (i.e. which DEATH output drives each
# electrode, from 0 -> 29)
physical_electrode_transform = np.array([0,4,8,2,  6,10,14,18,  22,26,30,16,  20,24,13,
                                         1,5,9,3,  7,11,15,19,  23,27,31,17,  21,25,29])

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

## DEATH channel max voltage outputs
max_elec_voltages = np.zeros(30)+9.0
max_death_voltages = max_elec_voltages[dac_channel_transform]

min_elec_voltages = -max_elec_voltages
min_death_voltages = -max_death_voltages

## Maximum number of samples the DEATH RAMs can hold
max_death_samples = 16384

## Electrode starts and ends in um, ordered from Electrode 0 -> 29
electrode_coords = np.array([[-2535,-1535],[-1515,-1015],[-995,-695],[-675,-520],[-500,-345],[-325,-170],[-150,150],[170,325],[345,500],[520,675],[695,995],[1015,1515],[1535,2535],[-2535,-1535],[-1515,-1015],[-995,-695],[-675,-520],[-500,-345],[-325,-170],[-150,150],[170,325],[345,500],[520,675],[695,995],[1015,1515],[1535,2535]])

## Utility functions
def erfspace(a, b, npts, erf_scaling=2.5):
    slope = b-a
    erf_y = sstat.norm.cdf(np.linspace(-erf_scaling, erf_scaling, npts))
    erf_y_slope = erf_y[-1]-erf_y[0]
    vout_zc = erf_y*slope/erf_y_slope # scale slope
    return vout_zc + a - vout_zc[0] # shift range

def vlinspace(start_vec, end_vec, npts, lin_fn = np.linspace):
    """ Linspace on column vectors specifying the starts and ends"""
    assert start_vec.shape[1] == end_vec.shape[1] == 1, "Need to input column vectors"
    return np.vstack(list(lin_fn(sv, ev, npts) for sv, ev in zip(start_vec, end_vec)))

def find_wells_from_samples(samples, roi_centre, roi_width):
    # Convenience function to avoid having to generate a WavPotential
    # class for every sample (uses identical code)
    # TODO: extend arguments, get WavPotential to use this too
    potential = np.dot(trap_mom.potentials[:,:len(physical_electrode_transform)],
                       samples[physical_electrode_transform])
    return find_wells(potential, trap_mom.transport_axis, mass_Ca, mode='precise',
                      roi_centre=roi_centre, roi_width=roi_width)

def find_wells(potential, z_axis, ion_mass, mode='quick', smoothing_ratio=80, polyfit_ratio=60, freq_threshold = 10*kHz, roi_centre=0*um, roi_width=2356*um):
    """For a given potential, return the location of the potential minima, their offsets and curvatures.
    potential: spatial potential vector
    mode: 'quick' or 'precise'.
    smoothing_ratio: fraction of total length of potentials vector to smoothe over (not used atm?!)
    polyfit_ratio: fraction of total length of potentials vector to fit a polynomial to
    freq_threshold: Minimum trapping frequency of the wells to store"""

    assert mode is 'quick' or mode is 'precise', "Input argument 'mode' only supports mode='quick' and mode='precise'"
    assert len(potential) == len(z_axis), "Potential and z axis inputs are of different lengths"

    # Extract potential within region of interest        
    roi_l = roi_centre - roi_width
    roi_r = roi_centre + roi_width
    roi_idx = (roi_l <= z_axis) & (z_axis <= roi_r)
    pot = np.ravel(potential[roi_idx])
    trap_axis_roi = z_axis[roi_idx]
    trap_axis_idx = np.arange(z_axis.shape[0])[roi_idx]
            
    pot_resolution = z_axis[1]-z_axis[0]

    potg2 = np.gradient(np.gradient(pot))
    # Ad-hoc filtering of the waveform potential with a top-hat
    # window 1/80th as big
    #potg2_filt = np.convolve(potg2,
    #                         np.ones(pot.size/smoothing_ratio)/(pot.size*smoothing_ratio),
    #                         mode='same')
    min_indices_candidates, = ssig.argrelextrema(pot, np.less_equal, order=20) # find relative minima. order=x to suppress detecting most spurious minima

    # Gather wells
    min_indices  = []
    offsets = []
    polys = []
    trap_freqs = []
    trap_locs = []
    for mi in min_indices_candidates: # Loop over candidates
        if mode is 'quick':
            # numerically evaluate from the raw data (noisy)
            grad = potg2[mi]/pot_resolution**2
            # grads = potg2_filt[mi]/pot_resolution**2
        elif mode is 'precise':
            ## Select region of interest
            idx1 = mi-5
            idx2 = mi+5
            # Prevent out of bound errors
            if idx1 < 0:
                idx1 = 0
            if idx2 > (pot.shape[0]-1):
                idx2 = pot.shape[0]-1

            # Fit quadratic to potential
            pfit = np.polyfit(trap_axis_roi[idx1:idx2], pot[idx1:idx2], 2)
            poly = np.poly1d(pfit)
            grad = 2*poly[2] # in eV

        # Only keep wells that are confining and above the threshold
        if grad > 0:
            freq = np.sqrt( electron_charge * grad / (ion_mass * atomic_mass_unit) ) /2/np.pi
            if freq > freq_threshold:
                min_indices.append(trap_axis_idx[mi])
                if mode is 'quick':
                    offsets.append(pot[mi])
                    trap_freqs.append(freq)
                    trap_locs.append(trap_axis_roi[mi])
                elif mode is 'precise':
                    polys.append(poly)
                    offsets.append(-poly[1]**2/4/poly[2]+poly[0])
                    trap_freqs.append(freq)
                    trap_locs.append(-poly[1]/2/poly[2])

    return {'min_indices':min_indices, 'offsets':offsets, 'freqs':trap_freqs, 'locs':trap_locs}
    
class Moments:
    """Spatial potential moments of the electrodes; used for calculations
    involving the trap"""
    def __init__(self,
                 moments_path = os.path.join(os.path.dirname(__file__), "moments_file", "DanielTrapMomentsTransport.mat"),
                 potential_path = os.path.join(os.path.dirname(__file__), "moments_file", "trap_exp.pickle"),
                #f_rf_drive = 115*MHz, # trap rf drive frequency (Hz)
                #v_rf = 415 # trap rf drive voltage (Volts)
                 ):
        
        #self.f_rf_drive = f_rf_drive
        #self.v_rf = v_rf
        
        self.load_trap_axis_potential_data(moments_path)
        self.load_3d_potential_data(potential_path)

    def load_trap_axis_potential_data(self, moments_path):
        """ Based on reduced_data_ludwig.m, reconstructed here.
        Extracts and stores the potentials along the trap axis due to the various electrodes,
        as well as the first few spatial derivatives with respect to the trap axis. """
        
        data = sio.loadmat(moments_path, struct_as_record=False)['DATA'][0][0]
        
        starting_shim_electrode = 30
        num_electrodes = 30 # Control electrodes DCCa0 to DCCa14 and DCCa0 to DCCa14
        num_shims = 20 # Shim electrodes DCS[a,b,c,d][1,2,3,4,5] e.g. DCSa1
        
        # The electrode moments store the potential of the respective electrode 
        # along the trap axis, as well as the first few derivatives. E.g.
        # V(z) = electrode_moments[:,0]
        # V'(z) = electrode_moments[:,1]
        # etc. up to V(5)(z) = electrode_moments[:,5]
        # However, for V'''(z) and higher derivatives the data becomes increasingly noisy.
        self.electrode_moments = []
        self.shim_moments = []
        
        for q in range(num_electrodes):
            self.electrode_moments.append(data.electrode[0,q].moments)

        for q in range(starting_shim_electrode, num_shims+starting_shim_electrode):
            self.shim_moments.append(data.electrode[0,q].moments)

        self.transport_axis = data.transport_axis.flatten()
        self.rf_pondpot = data.RF_pondpot # Potential due to RF electrodes along trap axis. Needs to be scaled with rf freq, voltage and ion mass.

        # More complete potential data
        # Organised as (number of z locations) * (number of electrodes) (different from Matlab)
        self.potentials = np.zeros([len(self.transport_axis), num_electrodes])
        for k in range(num_electrodes):
            self.potentials[:,k] = self.electrode_moments[k][:,0]

        # Higher-res potential data [don't need for now]
    
    def load_3d_potential_data(self, potential_path):
        """ Loads the potentials in 3d due to the individual trap electrodes as 
        obtained from simulations performed with the NIST BEM software. 
        The current data covers +-100um in the trap direction, and +-11um in the
        radial directions and is primarily used to calculate the radial frequencies
        and principal axes within the experimental zone. 
        """
        
        with open(potential_path, 'rb') as f:
            potentials, origin, spacing, dimensions, x, y, z, xx, yy, zz, coordinates = pickle.load(f)
            
        # Add up the contributions of the shim segments and add to dictionary
        V_DCsa = potentials['DCSa1'] + potentials['DCSa2'] + potentials['DCSa3'] + potentials['DCSa4'] + potentials['DCSa5']
        V_DCsb = potentials['DCSb1'] + potentials['DCSb2'] + potentials['DCSb3'] + potentials['DCSb4'] + potentials['DCSb5']
        V_DCsc = potentials['DCSc1'] + potentials['DCSc2'] + potentials['DCSc3'] + potentials['DCSc4'] + potentials['DCSc5']
        V_DCsd = potentials['DCSd1'] + potentials['DCSd2'] + potentials['DCSd3'] + potentials['DCSd4'] + potentials['DCSd5']

        potentials.update( {'DCSa' : V_DCsa} )
        potentials.update( {'DCSb' : V_DCsb} )
        potentials.update( {'DCSc' : V_DCsc} )
        potentials.update( {'DCSd' : V_DCsd} )
            
        # Define dummy class to use similar to a C struct in order to 
        # bundle the 3d potential data into a single object.
        class potentials_3d:
            pass
            
        pot3d = potentials_3d()            
        pot3d.potentials = potentials # dictionary containing the potentials of all the control & shim & rf electrodes
        pot3d.origin = origin # origin of the mesh
        pot3d.spacing = spacing # spacing of the mesh along the various axes
        pot3d.dimensions = dimensions # number of points in the mesh along the various axes
        pot3d.x = x # vector containing the points along a single axis
        pot3d.y = y # i.e. y = [-11, -9, ..., 9, 11]*um
        pot3d.z = z
        pot3d.xx = xx # vector with the x coordinates for all the mesh points, flattened
        pot3d.yy = yy # i.e. potentials['ElectrodeName'][ind] = V(xx[ind],yy[ind],zz[ind])
        pot3d.zz = zz
        pot3d.coordinates = coordinates # = [xx, yy, zz]
        pot3d.fit_coord3d = np.column_stack( (xx**2, yy**2, zz**2, xx*yy, xx*zz, yy*zz, xx, yy, zz, np.ones_like(zz)) ) # used for finding potential eigenaxes in 3d
        zz2d, yy2d = np.meshgrid(z,y) # coordinates for one slice in the radial plane
        yy2d = yy2d.flatten(order='F')
        zz2d = zz2d.flatten(order='F')
        pot3d.yy2d = yy2d
        pot3d.zz2d = zz2d
        pot3d.fit_coord2d = np.column_stack( (yy2d**2, zz2d**2, yy2d*zz2d, yy2d, zz2d, np.ones_like(zz2d)) ) # used for finding potential eigenaxes in 2d
        self.pot3d = pot3d
        
trap_mom = Moments() # Global trap moments

class WavDesired:
    """ Specifications describing potential wells to solve for"""
    def __init__(self,
                 potentials, # list of arrays; each array is a potential for a timestep; volts
                 roi_idx, # Element indices for global trap axis position array; dims must match potentials
                 Ts=100*ns, # slowdown of 0 -> 10 ns/step, slowdown of 30 (typical) -> (10*(30+1)) = 310 ns/step
                 mass=mass_Ca,
                 num_electrodes=30,
                 desc=None,
                 solver_weights=None):
        self.desc = desc
        self.potentials = potentials
        self.roi_idx = roi_idx
        self.Ts = Ts
        self.mass = mass
        self.num_electrodes = num_electrodes
        if desc:
            self.desc = desc
        else:
            self.desc = "No description specified"
        self.solver_weights = {
            # Cost function parameters
            'r0': 1e-4, # punishes deviations from r0_u_ss. Can be used to set default voltages for irrelevant electrodes.
            'r1': 1e-3, # punishes the first time derivative of u, thus limiting the slew rate
            'r2': 1e-4, # punishes the second time derivative of u, thus further enforcing smoothness

            # default voltage for the electrodes. any deviations from
            # this will be punished, weighted by r0 and r0_u_weights
            'r0_u_ss': np.ones(num_electrodes)*0.5, # default voltages for the electrodes
            'r0_u_weights': np.ones(num_electrodes) # use this to put different weights on outer electrodes
            }
        if solver_weights:
            # non-default solver parameters
            self.solver_weights.update(solver_weights)

    def plot(self, trap_axis, ax=None):
        """ ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        ax.plot(trap_axis[self.roi_idx]/um, self.potentials)
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        return ax

class WavDesiredWells(WavDesired):
    def __init__(self,
                 # array or list of arrays/lists (can be single-element) of each position as a fn of timestep
                 positions, 
                 freqs, # array, same dimensions as positions
                 offsets, # array, same dimensions as positions
                 desired_potential_params=None,
                 Ts=10*ns,
                 mass=mass_Ca,
                 num_electrodes=30,
                 desc=None,
                 solver_weights=None):
        
        potentials, roi_idx = self.desiredPotentials(positions, freqs, offsets, mass, desired_potential_params)
        
        super().__init__(potentials, roi_idx, Ts, mass, num_electrodes, desc, solver_weights)

    def desiredPotentials(self, pos, freq, off, mass, des_pot_parm=None):
        # lists as a function of timestep [STILL ASSUMING ONE WELL PER POTENTIAL]
        pot = []
        roi = []
        if des_pot_parm is not None:
            energy_threshold = des_pot_parm['energy_threshold']
        else:
            energy_threshold = 150*meV

        assert type(pos) is type(freq) is type(off), "Input types inconsistent"
        if type(pos) is list or tuple:
            # Construct 2D matrices from lists: columns of each are
            # the timesteps, rows of each are the discrete wells
            pos = np.vstack(pos).T
            freq = np.vstack(freq).T
            off = np.vstack(off).T

        for po, fr, of in zip(pos,freq,off): # iterate over timesteps
            assert len(po) is not 0, "Desired wells supplied in incorrect format: must be list of lists or 2D array"
            pot_l = np.empty(0)
            roi_l = np.empty(0, dtype='int')

            for po_l, fr_l, of_l in zip(po, fr, of): # iterate over discrete wells
                a = (2*np.pi*fr_l)**2 * (mass * atomic_mass_unit) / (2*electron_charge)
                v_desired = a * (trap_mom.transport_axis - po_l)**2 + of_l
                relevant_idx = np.argwhere(v_desired < of_l + energy_threshold).flatten()
                pot_l = np.hstack((pot_l, v_desired[relevant_idx])) # TODO: make more efficient
                roi_l = np.hstack((roi_l, relevant_idx))

            pot.append(pot_l)
            roi.append(roi_l)
        return pot, roi

    def plot(self, idx, trap_axis, ax=None):
        """ ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            #st()
        ax.plot(trap_axis[self.roi_idx[idx]]/um, self.potentials[idx])
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        return ax
    
class Waveform:
    """Waveform storage class. Convert an input list into a numpy array
    and store various details about it, or solve given a set of
    constraints and the global trap potentials.
    """
    def __init__(self, *args, **kwargs):
        if len(args) is 4:
            # Directly create Waveform from [desc, uid, samples, generated]
            self.desc = args[0]
            self.uid = args[1]
            self.generated = args[2]
            self.samples = np.array(args[3])
        elif isinstance(args[0],  WavDesired): # check if a child of WavDesired
			# Create waveform based on WavDesired by setting up and solving an optimal control problem
            wdp = args[0]
            raw_samples = self.solve_potentials(wdp) # ordered by electrode
            num_elec, num_timesteps = raw_samples.shape
            self.samples = np.zeros((num_elec+2,num_timesteps)) # Add two DEATH monitor channels
            self.samples[:,:] = raw_samples[list(abs(k) for k in dac_channel_transform),:] # Transform as required

            self.desc = wdp.desc
            self.set_new_uid()
            self.generated = ""
        else:
            assert False, "Need some arguments in __init__."

    def solve_potentials(self, wdp):
        """ Convert a desired set of potentials and ROIs into waveform samples"""
        # TODO: make this more flexible, i.e. arbitrary-size voltages
        # max_elec_voltages should be copied from config_local.h in ionpulse_sdk
        # max_elec_voltages = np.ones(wdp.num_electrodes)*9.0 
        # min_elec_voltages = -max_elec_voltages
        max_slew_rate = 5 / us # (volts / s)

        # Cost function parameters
        sw = wdp.solver_weights

        N = len(wdp.potentials) # timesteps

        ## Setup and solve optimisation problem
        uopt = cvy.Variable(wdp.num_electrodes, N)
        states = []

        # Global constraints

        # Penalise deviations from default voltage
        sw_r0_u_ss_m = np.tile(sw['r0_u_ss'], (N,1)).T # matrixized
        cost = sw['r0'] * cvy.sum_squares(sw['r0_u_weights'] * (uopt - sw_r0_u_ss_m))
                    
        # Absolute voltage constraints
        min_elec_voltages_m = np.tile(min_elec_voltages, (N,1)).T
        max_elec_voltages_m = np.tile(max_elec_voltages, (N,1)).T        
        constr = [min_elec_voltages_m <= uopt, uopt <= max_elec_voltages_m]

        # Absolute symmetry constraints
        constr += [uopt[:15,:] == uopt[15:,:]]

        # Approximate costs on first and second derivative of u with finite differences
        # Here, we use 2nd order approximations. For a table with coefficients see 
        # https://en.wikipedia.org/wiki/Finite_difference_coefficient

        assert (N < 2) or (N > 3), "Cannot have this number of timesteps, due to finite-diff approximations"
        if N > 3:
            # Middle: central finite-difference approx
            cost += sw['r1']*cvy.sum_squares(0.5*(uopt[:,2:]-uopt[:,:-2]) )
            cost += sw['r2']*cvy.sum_squares(uopt[:,2:] -2 * uopt[:,1:-1] + uopt[:,:-2])

            # Start: use forward finite difference approximation of derivatives
            cost += sw['r1']*cvy.sum_squares(-0.5*uopt[:,2] + 2*uopt[:,1] - 1.5*uopt[:,0])
            cost += sw['r2']*cvy.sum_squares(-uopt[:,3] + 4*uopt[:,2] - 5*uopt[:,1] + 2*uopt[:,0])

            # End: use backward finite difference approximation of derivatives
            cost += sw['r1']*cvy.sum_squares(1.5*uopt[:,-1] - 2*uopt[:,-2] + 0.5*uopt[:,-3])
            cost += sw['r2']*cvy.sum_squares(2*uopt[:,-1] - 5*uopt[:,-2] + 4*uopt[:,-3] - uopt[:,-4]) 

            # Slew rate constraints    
            constr += [-max_slew_rate*wdp.Ts <= (uopt[:,1:]-uopt[:,:-1]), (uopt[:,1:]-uopt[:,:-1]) <= max_slew_rate*wdp.Ts]
        
        for kk, (pot, roi) in enumerate(zip(wdp.potentials, wdp.roi_idx)):
            # Cost term capturing how accurately we generate the desired potential well
            # (could also vectorise it like the above, but the ROI
            # indices tend to vary in length between timesteps)
            cost += cvy.sum_squares(trap_mom.potentials[roi, :]*uopt[:,kk] - pot)

        states.append( cvy.Problem(cvy.Minimize(cost), constr) )

        # ECOS is faster than CVXOPT, but can crash for larger problems
        prob = sum(states)
        prob.solve(solver=global_solver, verbose=True)

        if False:
            # DEBUGGING ONLY
            plt.plot(trap_mom.transport_axis, trap_mom.potentials*uopt.value[:,0])
            plt.plot(trap_mom.transport_axis[wdp.roi_idx[0]], wdp.potentials[0],'--')
        
        return uopt.value

    def set_new_uid(self):
        self.uid = np.random.randint(0, 2**31)

    def voltage_limits_exceeded(self):
        for column in self.samples.T:
            if np.any(column > max_death_voltages) or np.any(column < min_death_voltages):
                return True
        return False
        
class WavPotential:
    """ Class for calculating, analyzing and visualizing the potentials resulting
    from the electrodes in both 1d (i.e. along the trap axis), 2d (i.e in the radial plane)
    and 3d (i.e. along the axial and radial directions). Generally to be used for analysing 
    and plotting existing waveforms"""
    def __init__(self, waveform, ion_mass=mass_Ca, rf_v=415, rf_freq=115, shim_alpha=0, shim_beta=0):
        
        ## Load relevant electrodes and reorder as needed
        # Warning: This is not very robust at the moment! Beware when modifying
        # physical_electrode_transform. Because we probably won't have to change
        # it in a long time, this is implemented sloppily for now.
        mom_trunc = trap_mom.potentials[:,:len(physical_electrode_transform)]
        waveform_samples_trunc = waveform.samples[physical_electrode_transform,:]

        # Assign arguments
        self.waveform_samples = waveform_samples_trunc
        self.potentials = np.dot(mom_trunc, waveform_samples_trunc) # Potentials along trap axis for all timesteps, [location_idx,timestep_idx]
        self.trap_axis = trap_mom.transport_axis
        self.ion_mass = ion_mass # (amu)
        self.rf_v = rf_v # (Volts)
        self.rf_freq = rf_freq # (MHz) (!!)
        self.shim_alpha = shim_alpha # (Volts)
        self.shim_beta = shim_beta # (Volts)

    ### Functions for analyzing/plotting the potential along the trap axis

    def plot(self, ax=None):
        """ Plot the whole waveform.
        ax: Matplotlib axes """
        trap_axis_spacing = self.trap_axis[1]-self.trap_axis[0]
        # Since plot is showing quadrilaterals
        trap_axis_pts = np.append(self.trap_axis, self.trap_axis[-1] + trap_axis_spacing) \
                        - trap_axis_spacing/2
        trap_axis_pts *= 1000 # convert from m to um
        px, py = np.meshgrid(np.arange(self.potentials.shape[1]+1), trap_axis_pts)

        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        pcm = ax.pcolormesh(px, py, self.potentials, cmap='gray')
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
            ax = fig.add_subplot(1,1,1)
        ax.plot(self.trap_axis/um, self.potentials[:,idx])
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        return ax

    def plot_electrodes(self, idx, ax=None):
        """ Plot the potential along the trap axis due to individual electrodes.
        idx: index of the electrode(s) to plot
        ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        
        for id in [idx]:
            ax.plot(trap_mom.potentials[:,id])

    def plot_range_of_wfms(self, timesteps, ax=None):
        """ Plot the potential along the trap axis at various timesteps.
        timesteps: int or iterable. Int: Number of timesteps, Iterable: desired timesteps
        ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        if type(timesteps) is int:
            # integer, specifying number of points
            idces = np.linspace(0, self.potentials.shape[1]-1, timesteps, dtype='int')
        else:
            # iterable
            idces = timesteps
        for idx in idces:
            ax.plot(self.trap_axis/um, self.potentials[:,idx])
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        return ax

    def animate_wfm(self, decimation=10):
        Writer = anim.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist="vnegnev"), bitrate=1800)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim([-2350,2350])
        ax.set_ylim([-4,4])
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        
        line, = ax.plot(self.trap_axis/um, self.potentials[:,0])
        def update(data):
            line.set_ydata(data)
            return line

        def data_gen():
            for pot in self.potentials.T[::decimation]:
                yield pot

        im_ani = anim.FuncAnimation(fig, update, data_gen, interval=30) # interval: ms between new frames
        plt.show()

    def find_wells(self, time_idx, mode='quick', smoothing_ratio=80, polyfit_ratio=60, freq_threshold=10*kHz, roi_centre=0*um, roi_width=2356*um):
        return find_wells(self.potentials[:,time_idx], self.trap_axis, self.ion_mass, mode, smoothing_ratio, polyfit_ratio, freq_threshold, roi_centre, roi_width)

    ### Functions for analyzing/plotting the potential in 2d (radials), and 3d (axial+radials)
        
    def add_potentials_2d(self, time_idx):
        """Calculates the potential due to all the electrodes (i.e. RF, control & shims) in the radial plane at the position of the axial well minimum."""
        
        ## Find relevant plane
        
        # Find minimum along trap axis
        axial_wells = self.find_wells(time_idx,mode='quick',roi_centre=0,roi_width=100*um)
        
        assert len(axial_wells['locs']) > 0, "Found no trapping well in -100um to +100um"
        assert len(axial_wells['locs']) < 2, "Found more than one trapping well in -100um to +100um"
        
        r0_x = axial_wells['locs'][0]
        axial_freq = axial_wells['freqs'][0]

        x_idx = ( np.abs(trap_mom.pot3d.x - r0_x) ).argmin()
        pot_slice_ind = np.arange(41*12*12).reshape(41,12,12,order='F')[x_idx,:,:] # relevant indices of flattened array
        
        ## Add potentials 
        
        # Add potential due to RF electrodes
        rf_scaling = self.rf_v**2/(self.ion_mass*self.rf_freq**2)
        potential_2d = trap_mom.pot3d.potentials['RF_pondpot_1V1MHz1amu'][pot_slice_ind]*rf_scaling
    
        # Add potential due to DC control electrodes
        for el_idx in range(len(physical_electrode_transform)):
            # Get electrode name
            assert (0 <= el_idx) and (el_idx <= 29), "Electrode index out of bounds"
            if el_idx < 15:
                # Top electrode, thus DCCc (see comment at the top)
                electrode_name = 'DCCc' + str(el_idx)
            else:
                # Bottom electrode, thus DCCa
                electrode_name = 'DCCa' + str(el_idx-15)
    
            potential_2d += self.waveform_samples[el_idx, time_idx]*trap_mom.pot3d.potentials[electrode_name][pot_slice_ind]
    
        # Add potential due to shims
        Vsa = self.shim_alpha/4 + self.shim_beta/4;
        Vsb = -self.shim_alpha/4 + self.shim_beta/4;
        Vsc = self.shim_alpha/4 - self.shim_beta/4;
        Vsd = -self.shim_alpha/4 - self.shim_beta/4;
        
        potential_2d +=  Vsa*trap_mom.pot3d.potentials['DCSa'][pot_slice_ind] + \
                         Vsb*trap_mom.pot3d.potentials['DCSb'][pot_slice_ind] + \
                         Vsc*trap_mom.pot3d.potentials['DCSc'][pot_slice_ind] + \
                         Vsd*trap_mom.pot3d.potentials['DCSd'][pot_slice_ind]

        potential_2d = potential_2d.reshape(12,12)
        
        return potential_2d, r0_x, axial_freq
    
    def find_radials_2d(self, time_idx):
        """Calculates the axial and radial trap frequencies assuming the axial direction to be along the trap axis, thus
        reducing the calculations required by only having to fit the radials to the potential V(y,z) rather than V(x,y,z).
        Returns all three trap frequencies, trap centre coordinates and axes."""
        # See find_radials_3d for a detailed description.
        # Notation:
        # V(y,z) = ay^2 + bz^2 + c*yz + d*y + e*z + f
        
        V, r0_x, axial_freq = self.add_potentials_2d(time_idx)
        
        # Linear least squares fit
        p = np.linalg.lstsq(trap_mom.pot3d.fit_coord2d,V.flatten(order='F'))[0]
        a,b,c,d,e,f=p
        
        # Extract the trapping frequencies and corresponding axes
        A = np.array( [ [a, c/2], [c/2, b] ] )
        eigenvalues, radial_axes =  np.linalg.eig(A)        
        radial_freqs = np.sqrt(eigenvalues*2*electron_charge / (self.ion_mass*atomic_mass_unit) )/(2*np.pi)
        
        # Extract the trapping location by solving for the point where grad V = 0:
        A2 = np.array( [ [2*a, c], [c, 2*b]] )
        b2 = -p[3:5] # -[d; e; f]
        r0_yz = np.linalg.lstsq(A2,b2)[0] # Solve A2*r0 = b        
        
        # Overall offset
        offset = f
    
        # Combine axial + radial information
        omegas = np.zeros(3)
        omegas[0] = axial_freq
        omegas[1:3] = radial_freqs
        if any(w < 0 for w in omegas):
            warnings.warn('Potential is anti-confining.')
        
        axes = np.zeros( (3,3) )
        axes[0,0] = 1 # axial eigenvector comes first
        if radial_axes[1,0] < 0: # align both radial eigenvectors in the upper half plane
            radial_axes[:,0] = -radial_axes[:,0]
        if radial_axes[1,1] < 0:
            radial_axes[:,1] = -radial_axes[:,1]
        axes[1:,1:] = radial_axes
        r0 = np.zeros(3)
        r0[0] = r0_x
        r0[1:3] = r0_yz
        
        return omegas, axes, r0, offset, V
        
    def add_potentials_3d(self, time_idx):
        """Calculates the 3d potential in the experimental zone of the trap due to all the electrodes (i.e. RF, control & shims) """
        
        # Add potential due to RF electrodes
        rf_scaling = self.rf_v**2/(self.ion_mass*self.rf_freq**2)
        potential_3d = trap_mom.pot3d.potentials['RF_pondpot_1V1MHz1amu']*rf_scaling
        
        # Add potential due to DC control electrodes
        for el_idx in range(len(physical_electrode_transform)):
            # Get electrode name
            assert (0 <= el_idx) and (el_idx <= 29), "Electrode index out of bounds"
            if el_idx < 15:
                # Top electrode, thus DCCc (see comment at the top)
                electrode_name = 'DCCc' + str(el_idx)
            else:
                # Bottom electrode, thus DCCa
                electrode_name = 'DCCa' + str(el_idx-15)

            potential_3d += self.waveform_samples[el_idx, time_idx]*trap_mom.pot3d.potentials[electrode_name]

        # Add potential due to shims
        Vsa = self.shim_alpha/4 + self.shim_beta/4;
        Vsb = -self.shim_alpha/4 + self.shim_beta/4;
        Vsc = self.shim_alpha/4 - self.shim_beta/4;
        Vsd = -self.shim_alpha/4 - self.shim_beta/4;
        
        potential_3d +=  Vsa*trap_mom.pot3d.potentials['DCSa'] + \
                         Vsb*trap_mom.pot3d.potentials['DCSb'] + \
                         Vsc*trap_mom.pot3d.potentials['DCSc'] + \
                         Vsd*trap_mom.pot3d.potentials['DCSd']
        
        return potential_3d
        
    def find_radials_3d(self,time_idx):
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
    
        V = self.add_potentials_3d(time_idx)
        
        # Linear least squares fit
        p = np.linalg.lstsq(trap_mom.pot3d.fit_coord3d,V)[0]
        a,b,c,d,e,f,g,h,i,j = p
        
        # Extract the trapping frequencies and corresponding axes:
        # Idea: Expanding (1), we get V(r) = (r-r0)'*A*(r-r0) = r'*A*r + ... 
        # The term r'*A*r corresponds to the terms
        # a*x^2 + b*y^2 + c*z^2 + d*xy + e*xz + f*yz from (2)
        # We can thus read off A from (a, ... ,f)
        # Calculating the eigenvalues and eigenvectors of A then gives us the trap
        # strengths and their associated axes.
        A = np.array( [ [a, d/2, e/2], [d/2, b, f/2 ], [e/2, f/2, c] ] )
        eigenvalues, axes =  np.linalg.eig(A) # each column of axes corresponds to one eigenvector
        omegas = np.sqrt(eigenvalues*2*electron_charge / (self.ion_mass*atomic_mass_unit) )/(2*np.pi) # Freq in Hz
        
        if any(w < 0 for w in omegas):
            warnings.warn('Potential is anti-confining.')        
        
        # Extract the trapping location by solving for the point where grad V = 0:
        # dV/dx = 2*a*x + d*y + e*z + g = 0
        # dV/dy = d*x + 2*b*y + f*z + h = 0
        # dV/dz = e*x + f*y + 2*c*z + i = 0
        # A2 = [2a d e; d 2b f; e f 2c], b = -[g; h; i]
        # A2*[x;y;z] = b2    
        A2 = np.array( [ [2*a, d, e], [d, 2*b, f], [e, f, 2*c] ] )
        b2 = -p[6:9] # h i j
        r0 = np.linalg.lstsq(A2,b2)[0] # Solve Ax = b
        
        # Overall offset
        offset = p[-1] # j

        # Sort the results such that omegas[0] and axes[:,0] correspond to the axial mode
        axial_mode_idx = axes[0,:].argmax()
        axial_mode_idx  = 0
        if axial_mode_idx != 0:
            if axial_mode_idx == 1:    
                permutation = np.array([axial_mode_idx,0,2],dtype='int')    
            elif axial_mode_idx == 2:
                permutation = np.array([axial_mode_idx,0,1],dtype='int')
            # Put axial first                
            omegas = omegas[permutation]
            axes = axes[:,permutation]
    
        return omegas, axes, r0, offset, V
        
    def plot_radials(self,time_idx, ax=None, mode='3d'):
        """ Plots the potential in the radial plane together with the radial directions,
        well centre position, and all frequencies.""" 
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        if mode == '3d':
            omegas, axes, r0, offset, V_3d = self.find_radials_3d(time_idx)
            # Pick required 2d slice from 3d potential:
            x_min_idx = ( np.abs(trap_mom.pot3d.x-r0[0]) ).argmin()
            V = V_3d.reshape((41,12,12),order='F')[x_min_idx,:,:]            
        elif mode == '2d':
            omegas, axes, r0, offset, V = self.find_radials_2d(time_idx)
        else:
            assert False, "Input argument 'mode' only supports mode='2d' or mode='3d'."
        
        # Align the axes correctly with respect to the imshow function
        V = V.T
        V = np.flipud(V)
        
        # Plot
        res = ax.imshow(V, extent=[-12, 12, -12, 12], interpolation='none',cmap='viridis')
        cbar = plt.colorbar(res, fraction=0.046, pad=0.04) # Numbers ensure cbar has same size as plot
        ax.plot(r0[1]/um,r0[2]/um,'r.',markersize=10)
        soa =np.array( [ [r0[1]/um, r0[2]/um, axes[1,1],axes[2,1]], [r0[1]/um,r0[2]/um,axes[1,2],axes[2,2]] ])
        X0,Y0,XV,YV = zip(*soa)
        scalefactor = 7 # length of arrows
        ax.quiver(X0,Y0,XV,YV,scale_units='xy',scale=1/scalefactor,color='white')
        
        # Annotate plot with trap freqs. and origin of well
        ax.text(r0[1]/um + 2, r0[2]/um - 0.5, '{:.2f}'.format(omegas[0]/MHz) + ' MHz', color='white') # Axial freq
        ax.text(r0[1]/um + scalefactor*axes[1,1], r0[2]/um + scalefactor*axes[2,1], '{:.2f}'.format(omegas[1]/MHz) + ' MHz', color='white') # Radial 1
        ax.text(r0[1]/um + scalefactor*axes[1,2], r0[2]/um + scalefactor*axes[2,2], '{:.2f}'.format(omegas[2]/MHz) + ' MHz', color='white') # Radial 2
        ax.text(r0[1]/um - 4, r0[2]/um - 5, 'x0 = ' + '{:.2f}'.format(r0[0]/um) + ' um\n' + 'y0 = ' + '{:.2f}'.format(r0[1]/um) + ' um\n' + 'z0 = ' + '{:.2f}'.format(r0[2]/um) + ' um', color='white') # Centre locations
                
        # Format plot
        ax.set_xlabel('y (um)')
        ax.set_ylabel('z (um)')
        plt.xticks(trap_mom.pot3d.y/um)
        plt.yticks(trap_mom.pot3d.z/um)
        ax.set_xlim([-12,12])
        ax.set_ylim([-12,12])
        cbar.set_label('Potential (V)')
        ax.set_title(mode +' analysis of the radials')          


        

# TODO: RO: Eliminate calculate_potentials?! Seems to be a glorified wrapper for
# the WavPotential init function.
def calculate_potentials(moments, waveform,
                         real_electrode_idxes=physical_electrode_transform,
                         ):
    """Multiplies the moments matrix by the waveform matrix
    (with suitable transformation based on real_electrode_idxes parameter)
    
    Note: watch out for scaling issues if real_electrode_indices is
    not equal to the number of electrodes
    
    moments: Moments class containing potential data
    waveform: Waveform class containing the voltage samples array

    """
    mom_trunc = moments.potentials[:,:len(real_electrode_idxes)]
    waveform_trunc = waveform.samples[real_electrode_idxes,:]
    
    # RO: TODO: Ca mass probably does not belong here!
    return WavPotential(waveform, mass_Ca)
    
class WaveformSet:
    """Waveform set handler, both for pre-generated JSON waveform files
    and waveform sets dynamically generated in Python"""
    def __init__(self, *args, **kwargs):
        """Can be used in several modes.

        The first is to generate a WaveformSet from an existing
        waveform file; the keyword argument waveform_file must have
        the filepath of the file to read.

        The second is to supply a list of existing Waveforms. These
        will be sorted by their names and checked for consistency.
        """

        if 'waveform_file' in kwargs.keys():
            # Create WaveformSet from existing JSON file
            with open(kwargs['waveform_file']) as fp:
                self.json_data = json.load(fp)
                waveform_num = len(self.json_data.keys())
                self.waveforms = [] # zero-indexed, unlike Matlab and Ionizer
                for k in range(1, waveform_num + 1):
                    jd = self.json_data['wav'+str(k)]
                    self.waveforms.append(Waveform(
                        jd['description'],
                        int(jd['uid'], 16),
                        jd['generated'],
                        jd['samples']
                    ))
                    
        elif type(args[0][0]) is Waveform:
            # Use existing list of Waveforms (no transfer of ownership for now!)
            # for k, wf in enumerate(args[0]):
            #     assert wf.desc is 'wav'+str(k+1), "Waveforms are not ordered. TODO: auto-order them"
            self.waveforms = args[0]
            
        else:
            # Nothing could be understood
            assert False, "Couldn't parse input args"

    def write(self, file_path):
        if os.path.isfile(file_path):
            warnings.warn("File "+file_path+" already exists. Overwriting...")
        with open(file_path, 'w') as fp:
            wfm_dict = {}
            total_samples_written = 0
            for k, wf in enumerate(self.waveforms):
                total_samples_written += wf.samples.shape[1]
                if total_samples_written > max_death_samples:
                    warnings.warn('Too many DEATH samples desired; truncating waveform file at Waveform ' + str(k+1))
                    break
                
                wfm_dict['wav'+str(k+1)] = {
                    'description':wf.desc,
                    'uid':hex(wf.uid), # cut off 0x to suit ionizer
                    'generated':wf.generated,
                    'samples':wf.samples.tolist()}
            json.dump(wfm_dict, fp, indent="", sort_keys=True)

    def get_waveform(self, num):
        """Return the waveform specified by a 1-indexed string or 0-indexed
        int. Accepts strings ('wav2') or ints (1).
        """
        if type(num) is str:
            idx = int(num[3:])-1
        elif type(num) is int:
            idx = num

        # assert idx >= 0, "Cannot access negative waveforms. Supply a 1-indexed string or 0-indexed int."
        return self.waveforms[idx]

    def find_waveform(self, name_str):
        """ Tries to find a waveform whose description partially or fully matches name_str """
        for w in self.waveforms:
            if name_str in w.desc:
                return w

if __name__ == "__main__":
    # Debugging stuff -- write unit tests from the below at some point
    # wfs.write("waveform_files/test_splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")
    
    radial_tests = True
    if radial_tests:
        ## Some tests showcasing the analysis of the radials
        # Generates a dummy static waveform and then analyzes it.
        
        wf_path = os.path.join("waveform_files", "radial_tests.dwc.json")
    
        # If file exists already, just load it to save time
        try:
            raise FileNotFoundError # uncomment to always regenerate file for debugging
            wfs_load = WaveformSet(waveform_file=wf_path)
            print("Loaded waveform ",wf_path)
        except FileNotFoundError:
            print("Generating waveform ",wf_path)
            
            local_weights = {'r0':1e-6,
                 'r0_u_weights':np.ones(30)*1e-4,
                 'r0_u_ss':np.ones(30)*8,
                 'r1':1e-6,'r2':1e-7}
                 
            local_potential_params={'energy_threshold':10*meV}
            
            def static_waveform(pos, freq, offs, wfm_desc):
                wdw = WavDesiredWells([pos*um],[freq*MHz],[offs*meV],            
                                      solver_weights=local_weights,
                                      desired_potential_params=local_potential_params,            
                                      desc=wfm_desc+", {:.3f} MHz, {:.1f} meV".format(freq, offs) )
                wf = Waveform(wdw)
                return wf
                
            wf_exp_static_13 = static_waveform(50, 1.8, 1000, "static")
            wf_list = [wf_exp_static_13]
            wfs_load = WaveformSet(wf_list)
            wfs_load.write(wf_path)
    
        # Analyze waveform
        WavPot = WavPotential( wfs_load.get_waveform(0),shim_beta = 0, shim_alpha = 0)
        
        WavPot.plot_radials(0, mode='2d')
        WavPot.plot_radials(0, mode='3d')
    
    if False:
        wfs = WaveformSet(waveform_file="waveform_files/splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")
        wfs.write("waveform_files/test2_splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")

    if False:
        # Generate loading waveform
        n_load = 1000
        wdp = WavDesiredWells(
            np.linspace(-1870,0,n_load)*um,
            np.linspace(1.1,1.3,n_load)*MHz,
            np.linspace(600,1000,n_load)*meV,
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
