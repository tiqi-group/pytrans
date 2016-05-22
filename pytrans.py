#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import scipy.io as sio
import scipy.signal as ssig
import cvxpy as cvy
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

# indices of electrodes 0->14 in the waveform files produced by the system right now
physical_electrode_transform = [0,4,8,2,6,10,14,12,  22,26,30,16,20,24,13]

class Moments:
    """Spatial potential moments of the electrodes; used for calculations
    involving the trap"""
    def __init__(self,
                 path="moments_file/DanielTrapMomentsTransport.mat"
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
        self.transport_axis = d.transport_axis
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

class Waveform:
    """Waveform storage class. Convert an input list into a numpy array
    and store various details about it.
    """
    def __init__(self, desc, uid, samples, generated):
        self.desc = desc
        self.uid = uid
        self.generated = generated
        self.samples = np.array(samples)
        self.channels, self.length = self.samples.shape

class WavPotential:
    """ Electric potential along the trap axis 

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
        elif mode is 'precise':
            # fit quadratics to the regions of interest
            offsets = []
            polys = []
            trap_freqs = []
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
            if False:
                plt.plot(self.trap_axis, pot,'r')            
                for p in polys:
                    plt.plot(self.trap_axis, p(self.trap_axis),'b')
                plt.show() 
        return min_indices, offsets, trap_freqs

class WaveformSet:
    """Waveform set handler, both for pre-generated JSON waveform files
    and waveform sets dynamically generated in Python"""
    def __init__(self, file_path=None):
        if file_path:
            """ Create WaveformSet from existing JSON file """
            with open(file_path) as fp:
                self.json_data = json.load(fp)
                waveform_num = len(self.json_data.keys())
                self.waveforms = [] # zero-indexed, unlike Matlab and Ionizer
                for k in range(1, waveform_num + 1):
                    jd = self.json_data['wav'+str(k)]
                    self.waveforms.append(Waveform(
                        desc = jd['description'],
                        uid = int(jd['uid'], 16),
                        generated = jd['generated'],
                        samples = jd['samples']
                    ))

    def write(self, file_path):
        with open(file_path, 'w') as fp:
            wfm_dict = {}
            for k, wf in enumerate(self.waveforms):
                wfm_dict['wav'+str(k+1)] = {
                    'description':wf.desc,
                    'uid':hex(wf.uid), # cut off 0x to suit ionizer
                    'generated':wf.generated,
                    'samples':wf.samples.tolist()}
            json.dump(wfm_dict, fp)

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
    # Debugging stuff
    wf = WaveformSet("waveform_files/splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")
#    wf.write("waveform_files/test_splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")
#    wf = WaveformSet("waveform_files/test_splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")
    wf.write("waveform_files/test2_splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json")
