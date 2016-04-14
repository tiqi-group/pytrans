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

# old indices of electrodes 0->14
# physical_electrode_transform = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

# indices of electrodes 0->14 in the waveform files produced by the system right now
physical_electrode_transform = [0,2,4,6,8,10,12,16,18,20,22,24,26,28,14]

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

    def plot_one_wfm(self, idx, ax=None):
        """ ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        ax.plot(self.trap_axis*1e6, self.potentials[:,idx])
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        return ax

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

        assert idx >= 0, "Cannot access negative waveforms. Supply a 1-indexed string or number."
        return self.waveforms[idx]

def calculate_potentials(moments, waveform,
                         real_electrode_idxes=physical_electrode_transform,
                         ):
    """ 
    Multiplies the moments matrix by the waveform matrix (with suitable truncation based on real_electrode_idxes parameter)
    moments: Moments class containing potential data
    waveform: Waveform class containing the voltage samples array
    """
    mom_trunc = moments.potentials[:,:len(real_electrode_idxes)]
    waveform_trunc = waveform.samples[real_electrode_idxes,:]
    
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
    stationary_comparison_with_old = False
    check_splitting_waveform = True
    
    mom = Moments()

    if (stationary_comparison_with_old):
        wf = WaveformFile('waveform_files/Ca_trans_load_open_Ca_Be_Transport_scan_freq_and_offset_pos_0_um.dwc.json')

        wf_load_54 = wf.get_waveform('wav54')
        pot_load_54 = calculate_potentials(mom, wf_load_54)

        wf_load_62 = wf.get_waveform('wav62')
        pot_load_62 = calculate_potentials(mom, wf_load_62)

        wf_load_104 = wf.get_waveform('wav104')
        pot_load_104 = calculate_potentials(mom, wf_load_104)    

        wf2 = WaveformFile('waveform_files/loading_and_constant_settings_Ts_620_2016_04_07_v02.dwc.json')

        wfms = (16, 25, 133) # or (17, 
        wfms_new = tuple(wf2.get_waveform(k) for k in wfms)
        pot_loads = tuple(calculate_potentials(mom, wf) for wf in wfms_new)

        def well_search():
            indices = []
            trap_freqs = []
            wfms = np.arange(500)
            for k in wfms:
                ind, _, tf = pot_load.find_wells(k)
                try:
                    indices.append(ind[np.argmax(tf)])
                except IndexError:
                    st()
                trap_freqs.append(tf.max())

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax = fig.add_subplot(111)
            st()
            ax.plot(np.array(wfms, dtype='float64'),
                    np.array(indices,dtype='float64'),
                    np.array(trap_freqs,dtype='float64'))
            plt.show()

        # well_search()
        test_wf = wf.get_waveform(9)
        plot_td_voltages(test_wf)
        # pot_load.find_wells(0)

        axa = pot_load_54.plot_one_wfm(0)
        pot_load_62.plot_one_wfm(0, axa)
        pot_load_104.plot_one_wfm(0, axa)    
        # pot_load.plot_one_wfm(-1, axa)

        for pl in pot_loads:
            pl.plot_one_wfm(0, axa)
        #pot_load2.plot_one_wfm(0, axa)
    #    pot_load2.plot_one_wfm(-1, axa)    

        # plt.plot(pot_load.potentials[:,990])
        # plt.show()
        # pot_load.plot()
        plt.show()
    
    if check_splitting_waveform:
        wf = WaveformFile('waveform_files/splitting_zone_Ts_620_vn_2016_04_14_v01.dwc.json')

        wf_all_sections = wf.get_waveform('wav8')
        pot_all_sections = calculate_potentials(mom, wf_all_sections)

        pot_all_sections.plot()
        plt.show()
