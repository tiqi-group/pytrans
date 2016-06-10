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
import pickle
import warnings
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

# Electrode mapping of the trap simulations: 
# DCCc(n) corresponds to T(n+1) = Index n
# DCCa(n) corresponds to B(n+1) = Index n+15, i.e. DCCa14 corresponds to B15 = Index 29
                  
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
# dac_channel_transform = [2,3,-0,-1,0,1] 
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
            
        # Define dummy class to use similar to a C struct in order to bundle 
        # the 3d potential data into a single object.
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
        pot3d.xx = xx # vector with the x coordinates for all the mesh points
        pot3d.yy = yy # i.e. potentials['ElectrodeName'][ind] = V(xx[ind],yy[ind],zz[ind])
        pot3d.zz = zz
        pot3d.coordinates = coordinates # = [xx, yy, zz]
        pot3d.fit_coord = np.column_stack( (xx**2, yy**2, zz**2, xx*yy, xx*zz, yy*zz, xx, yy, zz, np.ones_like(xx)) )
        
        self.pot3d = pot3d
        
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
    def __init__(self, waveform, ion_mass = 39.962591, rf_v = 415, rf_freq = 115, shim_alpha = 0, shim_beta = 0):
        """ potentials: (points along trap z axis) x (timesteps)
        ion mass: the ion's mass in AMU (for certain calculations)
        """
        
        # TODO: VN, please check if I handle this correctly.
        # Also, probably not very robust atm.
        mom_trunc = trap_mom.potentials[:,:len(physical_electrode_transform)]
        waveform_samples_trunc = waveform.samples[physical_electrode_transform,:]

        # Assign arguments
        self.waveform_samples = waveform_samples_trunc
        self.potentials = np.dot(mom_trunc, waveform_samples_trunc) # Potentials along trap axis for all timesteps
        self.trap_axis = trap_mom.transport_axis
        self.ion_mass = ion_mass # (amu)
        self.rf_v = rf_v # (Volts)
        self.rf_freq = rf_freq # (MHz)
        self.shim_alpha = shim_alpha # (Volts)
        self.shim_beta = shim_beta # (Volts)
        
        self.pot_resolution = self.trap_axis[1]-self.trap_axis[0]

    ### Functions analyzing/plotting the potential along the trap axis

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

    def find_wells(self, idx, mode='quick', smoothing_ratio=80, polyfit_ratio=60):
        """For a given timestep, return the location of the minima, their offsets and curvatures.
        idx: Desired timestep to analyze
        mode: 'quick' or 'precise'.
        smoothing_ratio: fraction of total length of potentials vector to smoothe over (not used atm?!)
        polyfit_ratio: fraction of total length of potentials vector to fit a polynomial to"""
        pot = self.potentials[:,idx]
        potg2 = np.gradient(np.gradient(pot))#self.pot_resolution**2
        # Ad-hoc filtering of the waveform potential with a top-hat
        # window 1/80th as big
        potg2_filt = np.convolve(potg2,
                                 np.ones(pot.size/smoothing_ratio)/(pot.size*smoothing_ratio),
                                 mode='same')
        min_indices, = ssig.argrelmin(pot, order=20) # relative minima. order=x to suppress detecting spurious minima
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
                # Prevent out of bound errors
                if idx1 < 0:
                    idx1 = 0
                if idx2 > (pot.shape[0]-1):
                    idx2 = pot.shape[0]-1
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

    ### Functions analyzing/plotting the potential in 2d (radials), and 3d (axial+radials)
    
    def find_radials_2d(self, ):
        """Calculate the two trapping frequencies and orthonormal directions given by the potential v(y,z)"""

        # See find_radials_3d for a detailed description.
        # Notation:
        # V(y,z) = ay^2 + bz^2 + c*yz + d*y + e*z + f
        
        
    def add_potentials_3d(self, time_idx):
        """Calculates the potential in the experimental zone of the trap due to all the electrodes (RF, control, shims) """
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
        # pot3d.fit_coord = np.column_stack( x**2, y**2, z**2, x*y, x*z, y*z, x, y, z, np.ones_like(x)) )
        p = np.linalg.lstsq(trap_mom.pot3d.fit_coord,V)[0]
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
    
        return omegas, axes, r0, offset, V
        
    def plot_radials_3d(self,time_idx, ax=None):
        """ WASD ESDF """ # TODO: 
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        
        omegas_3d, axes_3d, r0_3d, offset_3d, V = self.find_radials_3d(time_idx)
        
        V = V.reshape(41,12,12,order='F')
        V = V[20,:,:] # TODO: Look up index
        
        
        V = V.T
        V = np.flipud(V)
        
        res = ax.imshow(V, extent=[-12, 12, -12, 12], interpolation='none')
        plt.colorbar(res, fraction=0.046, pad=0.04)
        ax.plot(r0_3d[1]/um,r0_3d[2]/um,'r.',markersize=20)
        soa =np.array( [ [r0_3d[1]/um, r0_3d[2]/um, axes_3d[1,1],axes_3d[2,1]], [r0_3d[1]/um,r0_3d[2]/um,axes_3d[1,2],axes_3d[2,2]] ])
        X,Y,U,V = zip(*soa)
        ax.quiver(X,Y,U,V,scale_units='xy',scale=.1)
        
        ax.set_xlabel('y (um)')
        ax.set_ylabel('z (um)')
        plt.xticks(trap_mom.pot3d.y/um)
        plt.yticks(trap_mom.pot3d.z/um)
        ax.set_xlim([-12,12])
        ax.set_ylim([-12,12])

# RO: TODO: Eliminate calculate_potentials?! Seems to be a glorified wrapper for
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
    return WavPotential(waveform.samples, np.dot(mom_trunc, waveform_trunc), moments.transport_axis, 39.962591)
    
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
        pot_test = calculate_potentials(trap_mom, wfs.get_waveform(1))
        pot_test.plot_one_wfm(0)
        pot_test.plot_one_wfm(-1)        
        plt.show()
