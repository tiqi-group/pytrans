#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mpl_toolkits.mplot3d
import scipy.io as sio
import scipy.signal as ssig
import cvxpy as cvy
import os
import pdb
st = pdb.set_trace

# Unit definitions, all in SI
electron_charge = 1.60217662e-19 # coulombs
atomic_mass_unit = 1.66053904e-27 # kg
um = 1e-6
us = 1e-6
ns = 1e-9
MHz = 1e6
kHz = 1e3
meV = 1e-3

# Indexing convention:
# Electrodes T1 = 0, T2 = 1, ... T15 = 14; B1 = 15, B2 = 16, ... B15 = 29
# DEATH channels (top-right downwards, leftwards; bottom DEATHs 0->15, top DEATHs 16->31

# indices of which electrode each DEATH output drives, from 0->31
dac_channel_transform = np.array([0, 15,3,18, 1,16,4,19,   2,17,5,20,-7,14, 6,21,
                                  11,26,7,22,12,27,8,23,  13,28,9,24,-22,29,10,25])


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
# dac_channel_transform = [2,4,-0,-1,0,1] 
# physical_electrode_transform = [4, 5, 0, 1]

## DEATH channel max voltage outputs
max_elec_voltages = np.zeros(30)+9.0
max_death_voltages = max_elec_voltages[dac_channel_transform]

min_elec_voltages = -max_elec_voltages
min_death_voltages = -max_death_voltages

## Electrode starts and ends in um, ordered from Electrode 0 -> 29
electrode_coords = np.array([[-2535,-1535],[-1515,-1015],[-995,-695],[-675,-520],[-500,-345],[-325,-170],[-150,150],[170,325],[345,500],[520,675],[695,995],[1015,1515],[1535,2535],[-2535,-1535],[-1515,-1015],[-995,-695],[-675,-520],[-500,-345],[-325,-170],[-150,150],[170,325],[345,500],[520,675],[695,995],[1015,1515],[1535,2535]])

## Utility functions

def vlinspace(start_vec, end_vec, npts, lin_fn = np.linspace):
    """ Linspace on column vectors specifying the starts and ends"""
    assert start_vec.shape[1] == end_vec.shape[1] == 1, "Need to input column vectors"
    return np.vstack(list(lin_fn(sv, ev, npts) for sv, ev in zip(start_vec, end_vec)))

class Moments:
    """Spatial potential moments of the electrodes; used for calculations
    involving the trap"""
    def __init__(self,
                 path=os.path.join(os.path.dirname(__file__), "moments_file", "DanielTrapMomentsTransport.mat")
                 ):
        self.data = sio.loadmat(path, struct_as_record=False)['DATA'][0][0]
        self.reduce_data()

    def reduce_data(self):
        """ Based on reduced_data_ludwig.m """
        starting_shim_electrode = 30
        num_electrodes = 30
        num_shims = 20
        self.electrode_moments = []
        self.shim_moments = []        
        
        for q in range(num_electrodes):
            self.electrode_moments.append(self.data.electrode[0,q].moments)

        for q in range(starting_shim_electrode, num_shims+starting_shim_electrode):
            self.shim_moments.append(self.data.electrode[0,q].moments)

        d = self.data
        self.transport_axis = d.transport_axis.flatten()
        self.rf_pondpot = d.RF_pondpot
        self.amu = d.amu[0][0]
        self.w_t = d.w_t[0][0]
        self.rf_v = d.RF_V[0][0]

        # More complete potential data
        # Organised as (number of z locations) * (number of electrodes) (different from Matlab)
        self.potentials = np.zeros([len(self.transport_axis), num_electrodes])
        for k in range(num_electrodes):
            self.potentials[:,k] = self.electrode_moments[k][:,0]

        # Higher-res potential data [don't need for now]

trap_mom = Moments() # Global trap moments

class WavDesired:
    """ Specifications describing potential wells to solve for"""
    def __init__(self,
                 potentials, # list of arrays; each array is a potential for a timestep; volts
                 roi_idx, # Element indices for global trap axis position array; dims must match potentials
                 Ts=100*ns, # slowdown of 0 -> 10 ns/step, slowdown of 30 (typical) -> (10*(30+1)) = 310 ns/step
                 mass=39.962591,
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
                 Ts=100*ns,
                 mass=39.962591, # AMU
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
        if type(pos) is list:
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

        N = len(wdp.potentials)

        ## Setup and solve optimisation problem
        uopt = cvy.Variable(wdp.num_electrodes, N)
        states = [] # lists?

        for kk, (pot, roi) in enumerate(zip(wdp.potentials, wdp.roi_idx)):
            # Cost term capturing how accurately we generate the desired potential well            
            cost = cvy.sum_squares(trap_mom.potentials[roi, :]*uopt[:,kk] - pot)
            cost += sw['r0'] * cvy.sum_squares(sw['r0_u_weights'] * (uopt[:,kk] - sw['r0_u_ss']))
            
            # Absolute voltage constraints
            constr = [min_elec_voltages <= uopt[:,kk], uopt[:,kk] <= max_elec_voltages]

            # Absolute symmetry constraints
            for m in range(15):
                # symmetry constraints for electrode pairs
                constr += [uopt[m,kk] == uopt[m+15,kk]]

            assert (N < 2) or (N > 3), "Cannot have this number of timesteps, due to finite-diff approximations"
            if N > 3: # time-dependent constraints require at least 4 samples

                # Approximate costs on first and second derivative of u with finite differences
                # Here, we use 2nd order approximations. For a table with coefficients see 
                # https://en.wikipedia.org/wiki/Finite_difference_coefficient
                if ( kk != 0 and kk != N-1 ):
                    # Middle: Use central finite difference approximation of derivatives
                    cost += sw['r1']*cvy.sum_squares(0.5*(uopt[:,kk+1]-uopt[:,kk-1]) )
                    cost += sw['r2']*cvy.sum_squares(uopt[:,kk+1] - 2*uopt[:,kk] + uopt[:,kk-1])
                elif kk == 0:
                    # Start: Use forward finite difference approximation of derivatives
                    cost += sw['r1']*cvy.sum_squares(-0.5*uopt[:,kk+2] + 2*uopt[:,kk+1] - 1.5*uopt[:,kk])
                    cost += sw['r2']*cvy.sum_squares(-uopt[:,kk+3] + 4*uopt[:,kk+2] - 5*uopt[:,kk+1] + 2*uopt[:,kk])
                elif kk == N-1: 
                    # End: Use backward finite difference approximation of derivatives
                    cost += sw['r1']*cvy.sum_squares(1.5*uopt[:,kk] - 2*uopt[:,kk-1] + 0.5*uopt[:,kk-2])
                    cost += sw['r2']*cvy.sum_squares(2*uopt[:,kk] - 5*uopt[:,kk-1] + 4*uopt[:,kk-2] - uopt[:,kk-3]) 
                # Slew rate constraints    
                if (kk != N-1):
                    constr += [ -max_slew_rate*wdp.Ts <= uopt[:,kk+1] - uopt[:,kk] , uopt[:,kk+1] - uopt[:,kk] <= max_slew_rate*wdp.Ts ]

            states.append( cvy.Problem(cvy.Minimize(cost), constr) )

        # ECOS is faster than CVXOPT, but can crash for larger problems
        prob = sum(states)
        prob.solve(solver='ECOS', verbose=False)

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
    """ Electric potential along the trap axis (after solver!)
    Generally to be used for analysing and plotting existing waveforms
    TODO: include radial aspects too"""
    def __init__(self, potentials, trap_axis, ion_mass):
        """ potentials: (points along trap z axis) x (timesteps)
        trap_axis: physical coordinates of potentials along trap axis
        ion mass: the ion's mass in AMU (for certain calculations)
        """
        self.potentials = potentials
        self.trap_axis = trap_axis
        self.ion_mass = ion_mass
        self.pot_resolution = self.trap_axis[1]-self.trap_axis[0]

    def plot(self, ax=None):
        """ ax: Matplotlib axes """
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
        """ ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        ax.plot(self.trap_axis/um, self.potentials[:,idx])
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        return ax

    def plot_electrodes(self, idx, ax=None):
        """ ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        # finish later

    def plot_range_of_wfms(self, timesteps, ax=None):
        """ ax: Matplotlib axes """
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

    def find_wells(self, wfm_idx, mode='quick', smoothing_ratio=80, polyfit_ratio=60):
        """For a given waveform index, return the minima and their
        curvatures.
        smoothing_ratio: fraction of total length of potentials vector to smoothe over [not used?]
        polyfit_ratio: fraction of total length of potentials vector to fit a polynomial to"""
        pot = self.potentials[:,wfm_idx]
        potg2 = np.gradient(np.gradient(pot))#self.pot_resolution**2
        # Ad-hoc filtering of the waveform potential with a top-hat
        # window 1/80th as big
        potg2_filt = np.convolve(potg2,
                                 np.ones(pot.size/smoothing_ratio)/(pot.size*smoothing_ratio),
                                 mode='same')
        min_indices, = ssig.argrelmin(pot) # relative minima
        if mode is 'quick':
            # numerically evaluate from the raw data (noisy)
            offsets = pot[min_indices]
            # grads = potg2_filt[min_indices]/(self.pot_resolution**2)
            grads = potg2[min_indices]/(self.pot_resolution**2)
            #trap_freqs = np.sqrt(electron_charge * self.ion_mass / atomic_mass_unit * grads)/2/np.pi
            trap_freqs = np.sqrt(electron_charge * grads / (self.ion_mass * atomic_mass_unit))/2/np.pi
            trap_locs = trap_mom.transport_axis[min_indices]
        elif mode is 'precise':
            # fit quadratics to the regions of interest
            offsets = []
            polys = []
            trap_freqs = []
            trap_locs = []
            for mi in min_indices:
                idx1 = mi-pot.size//(polyfit_ratio*2)
                idx2 = mi+pot.size//(polyfit_ratio*2)
                pot_roi = pot[idx1:idx2]
                pot_z = self.trap_axis[idx1:idx2].flatten()
                #pfit = np.polyfit(pot_roi, pot_z, 2)
                pfit = np.polyfit(pot_z, pot_roi, 2)
                poly = np.poly1d(pfit)
                polys.append(poly)
                offsets.append(-poly[1]**2/4/poly[2]+poly[0])
                grad = poly.deriv().deriv() # in eV
                trap_freqs.append(np.sqrt(electron_charge * grad / (self.ion_mass * atomic_mass_unit))/2/np.pi)
                trap_locs.append(-poly[1]/2/poly[2])
            if False:
                plt.plot(self.trap_axis, pot,'r')            
                for p in polys:
                    plt.plot(self.trap_axis, p(self.trap_axis),'b')
                plt.show() 
        return {'min_indices':min_indices, 'offsets':offsets, 'freqs':trap_freqs, 'locs':trap_locs}

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
    
    return WavPotential(np.dot(mom_trunc, waveform_trunc), moments.transport_axis, 39.962591)
    
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
        with open(file_path, 'w') as fp:
            wfm_dict = {}
            for k, wf in enumerate(self.waveforms):
                wfm_dict['wav'+str(k+1)] = {
                    'description':wf.desc,
                    'uid':hex(wf.uid), # cut off 0x to suit ionizer
                    'generated':wf.generated,
                    'samples':wf.samples.tolist()}
            json.dump(wfm_dict, fp, indent="", sort_keys=True)

    def get_waveform(self, num):
        """ Return the 1-indexed waveform. Accepts strings ('wav2') or
        ints (2)."""
        if type(num) is str:
            idx = int(num[3:])-1
        elif type(num) is int:
            idx = num-1

        assert idx >= 0, "Cannot access negative waveforms. Supply a 1-indexed string or number."
        return self.waveforms[idx]        

if __name__ == "__main__":
    # Debugging stuff -- write unit tests from the below at some point
    # wfs.write("waveform_files/test_splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")
    if True:
        wfs = WaveformSet(waveform_file="waveform_files/splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")
        wfs.write("waveform_files/test2_splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")

    if True:
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

    if True:
        # Plot the above-generated waveform
        wfs = WaveformSet(waveform_file="waveform_files/loading_py_2016_05_23_v01.dwc.json")
        pot_test = calculate_potentials(trap_mom, wfs.get_waveform(1))
        pot_test.plot_one_wfm(0)
        pot_test.plot_one_wfm(-1)        
        plt.show()
    
    
