import json
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import scipy.io as sio
import scipy.signal as ssig
import pdb
st = pdb.set_trace

electron_charge = 1.60217662e-19 # coulombs
atomic_mass_unit = 1.66053904e-27 # kg

physical_electrodes = 15

class Moments:
    def __init__(self,
                 # path="/media/sf_Scratch/Waveform Generator 3D-trap/moments file/DanielTrapMomentsTransport.mat"
                 path = "c:/Scratch/wav_gen/moments_file/DanielTrapMomentsTransport.mat"
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
    def __init__(self, desc, uid, samples, generated):
        """ Waveform storage class. Convert the list into a numpy
        array and store various details about it. """
        self.desc = desc
        self.uid = uid
        self.generated = generated
        self.samples = np.array(samples)
        self.channels, self.length = self.samples.shape

class WavPotential:
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

    def find_wells(self, wfm_idx):
        """For a given waveform index, return the minima and their
        curvatures"""
        pot = self.potentials[:,wfm_idx]
        potg2 = np.gradient(np.gradient(pot))
        min_indices, = ssig.argrelmin(pot)
        offsets = pot[min_indices]
        grads = potg2[min_indices]/(self.pot_resolution**2)
        trap_freqs = np.sqrt(electron_charge * self.ion_mass / atomic_mass_unit * grads)/2/np.pi

        return min_indices, offsets, trap_freqs

class WaveformFile:
    def __init__(self, file_path):
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

    def get_waveform(self, num):
        """ Return the 1-indexed waveform. Accepts strings ('wav2') or
        ints (2)."""
        if type(num) is str:
            idx = int(num[3:])-1
        elif type(num) is int:
            idx = num-1

        return self.waveforms[idx]

def calculate_potentials(moments, waveform, real_electrodes=physical_electrodes):
    """ 
    Multiplies the moments matrix by the waveform matrix (with suitable truncation based on real_electrodes parameter)
    moments: Moments class containing potential data
    waveform: Waveform class containing the voltage samples array
    """
    mom_trunc = moments.potentials[:,:real_electrodes]
    waveform_trunc = waveform.samples[:real_electrodes,:]
    return WavPotential(np.dot(mom_trunc, waveform_trunc), moments.transport_axis, 39.962591)

def plot_td_voltages(waveform, electrodes_to_use=None, real_electrodes=physical_electrodes):
    """ Plot time-dependent voltages of a waveform w.r.t. electrodes as"""
    td_wfms = waveform.samples.T
    if electrodes_to_use:
        td_wfms = td_wfms[electrodes_to_use]
        leg = tuple(str(k+1) for k in electrodes_to_use)
    else:
        leg = tuple(str(k+1) for k in range(real_electrodes))
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(td_wfms)
    ax.legend(leg)
    plt.show()
    
if __name__ == "__main__":

    mom = Moments()
    
    # wf = WaveformFile('waveform_files/splitting_zone_Ts_70_vn_2016_01_29_v01.dwc.json')
    wf = WaveformFile('c:/scratch/wav_gen/waveforms_for_experiment/splitting_zone_Ts_620_vn_2016_03_16_v01.dwc.json')
    test_wf = wf.get_waveform(9)

    pot_test = calculate_potentials(mom, test_wf)

    def well_search():
        indices = []
        trap_freqs = []
        samples = np.arange(2000) # assume max number of samples in a waveform is 2000
        for k in samples:
            try:
                ind, _, tf = pot_test.find_wells(k)
                trap_freqs.append(tf.max())                
            except IndexError:
                samples = np.arange(k)
                break
            try:
                indices.append(ind[np.argmax(tf)])
            except IndexError:
                st()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax = fig.add_subplot(111)
        ax.plot(np.array(samples, dtype='float64'),
                np.array(indices,dtype='float64'),
                np.array(trap_freqs,dtype='float64'))
        plt.show()

    # well_search()
            
    # pot_test.find_wells(0)
    plot_td_voltages(test_wf)

    # Plot the potential of every 10th timestep
    plt.plot(pot_test.potentials[:,::10])
    plt.show()
    pot_test.plot()
    plt.show()
