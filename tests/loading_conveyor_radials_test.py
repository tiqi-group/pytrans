#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

local_weights = {'r0':1e-6,
                 'r0_u_weights':np.ones(30)*1e-4,
                 'r0_u_ss':np.ones(30)*8,
                 'r1':1e-6,'r2':1e-7}

local_potential_params={'energy_threshold':10*meV}

def plot_selection(pot):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    legends = []
    #pot.plot_range_of_wfms(np.linspace(400,800,100, dtype='int'))
    
    #pot.plot()
    pot.plot_one_wfm(0, ax)
    #pot.plot_electrodes(15)
    #pot.plot_range_of_wfms(5, ax)
    
    # for k in range(0,201,10):        
    #     pot.plot_one_wfm(k, ax)
    #     legends.append(str(k))

    plt.legend(legends)
    plt.show()

    # pot.plot_range_of_wfms(np.linspace(930,1000,20, dtype='int'))
    # pot.plot_range_of_wfms(np.linspace(149,199,50, dtype='int'))
    #plt.show()

def static_waveform(pos, freq, offs, wfm_desc):
    wdw = WavDesiredWells([pos*um],[freq*MHz],[offs*meV],

                          solver_weights=local_weights,
                          desired_potential_params=local_potential_params,

                          desc=wfm_desc+", {:.3f} MHz, {:.1f} meV".format(freq, offs)
    )
    wf = Waveform(wdw)
    return wf
    
def transport_waveform(pos, freq, offs, timesteps, wfm_desc, linspace_fn=np.linspace):
    wdw = WavDesiredWells(
        [linspace_fn(pos[0], pos[1], timesteps)*um],
        [linspace_fn(freq[0], freq[1], timesteps)*MHz],
        [linspace_fn(offs[0], offs[1], timesteps)*meV],

        solver_weights=local_weights,
        desired_potential_params=local_potential_params,

        desc=wfm_desc+", {:.3f}->{:.3f} MHz, {:.1f}->{:.1f} meV".format(freq[0], freq[1], offs[0], offs[1])
    )
    return Waveform(wdw)

def conveyor_waveform(pos, freq, offs, timesteps, wfm_desc, linspace_fn=np.linspace):
    pts_for_new_wfm = timesteps//5
    conveyor_timesteps = timesteps - pts_for_new_wfm
    offs_ramp_timesteps = (2*conveyor_timesteps)//3
    offs_const_timesteps = conveyor_timesteps - offs_ramp_timesteps
    # Generate the merging section
    wdw = WavDesiredWells(
        [linspace_fn(pos[0], pos[1], conveyor_timesteps)*um, np.ones(conveyor_timesteps)*pos[1]*um],
        [linspace_fn(freq[0], freq[1], conveyor_timesteps)*MHz, np.ones(conveyor_timesteps)*freq[1]*MHz],
        [np.hstack([
            linspace_fn(offs[0], offs[1], offs_ramp_timesteps),
            np.ones(offs_const_timesteps)*offs[1]])*meV, np.ones(conveyor_timesteps)*offs[1]*meV],

        solver_weights=local_weights,
        desired_potential_params=local_potential_params,

        desc=wfm_desc+" conveyor, {:.3f}->{:.3f} MHz, {:.1f}->{:.1f} meV".format(freq[0], freq[1], offs[0], offs[1])
    )
    # Generate the loading well recreation section
    wf = Waveform(wdw)

    if False:
        # Generate a manual waveform whose well goes down from 4200 meV to
        # the initial loading point, manually hacked-in (see potential
        # plots to diagnose level; highly sensitive to waveform solver
        final_loading_pot = 4200 #4230
        wf_rec_dw = WavDesiredWells(
            [np.ones(pts_for_new_wfm)*pos[0]*um, np.ones(pts_for_new_wfm)*pos[1]*um],
            [linspace_fn(0.1, freq[0], pts_for_new_wfm)*MHz, np.ones(pts_for_new_wfm)*freq[1]*MHz],
            [linspace_fn(final_loading_pot, offs[0], pts_for_new_wfm)*meV, np.ones(pts_for_new_wfm)*offs[1]*um],

            solver_weights=local_weights,
            desired_potential_params=local_potential_params,
            desc="")
        wf_rec = Waveform(wf_rec_dw)
        wf.samples = np.hstack([wf.samples, wf_rec.samples])            
    else:
        wf.samples = np.hstack([wf.samples,
                                vlinspace(wf.samples[:,[-1]], wf.samples[:,[0]], pts_for_new_wfm)])
    
    return wf

def reordering_waveform(pos, freq, offs, timesteps, push_v, twist_v, wfm_desc):
    # push_v: voltage by which to increase one side of electrodes
    # twist_v: voltage which is to be differentially applied to twist crystal
    # timesteps: total number of samples for full waveform
    wdw = WavDesiredWells([pos*um], [freq*MHz], [offs*meV],

                          solver_weights=local_weights,
                          desired_potential_params=local_potential_params,

                          desc=wfm_desc+", {:.3f} MHz, {:.1f} meV, push {:.2f} V, twist {:.2f} V".format(freq, offs, push_v, twist_v)
    )
    wf = Waveform(wdw)
    elec_start = wf.samples # vert. array

    def offset_voltages(elec_wfm, electrodes, offsets):
        # elec_wfm: vertical vector with electrode voltages
        # electrodes: array, list or int with electrodes to shift
        # offsets: array or int to shift electrodes by
        if type(offsets) is list:
            offsets = np.array([offsets]).T # to make it 2D
        elec_wfm2 = elec_wfm.copy()
        elec_wfm2[physical_electrode_transform[electrodes]] = elec_wfm[physical_electrode_transform[electrodes]] + offsets
        return elec_wfm2
    
    # Increase potentials on one side of trap, to 'push' out the crystal
    elec_push = offset_voltages(elec_start, [6,7,8], push_v)

    # Differentially twist electrode voltages after push
    elec_push_twist = offset_voltages(elec_push, [6, 8, 21, 23],
                                 [twist_v, -twist_v, -twist_v, twist_v])

    # Undo the push, keep the twist
    elec_twist = offset_voltages(elec_start, [6, 8, 21, 23],
                                 [twist_v, -twist_v, -twist_v, twist_v])
    segment_ts = timesteps//4
    wf.samples = np.hstack([vlinspace(elec_start, elec_push, segment_ts),                            
                            vlinspace(elec_push, elec_push_twist, segment_ts),
                            vlinspace(elec_push_twist, elec_twist, segment_ts),
                            # to avoid rounding changing the number of total samples
                            vlinspace(elec_twist, elec_start, timesteps-3*segment_ts)]) 
    
    return wf
    
def loading_conveyor(analyse_wfms=False):
    wf_path = os.path.join(os.pardir, "waveform_files", "loading_conveyor_2016_06_02_v01.dwc.json")

    # If file exists already, just load it to save time
    try:
        # raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ",wf_path)
        n_load = 1001
        n_freq_change = 200
        wf_load = transport_waveform(
            [-1870, 0], [0.7, 1.3], [600, 1000], n_load, "Load -> exp")
        wf_load_conveyor = conveyor_waveform(
            [-1870, 0], [0.7, 1.3], [600, 1000], n_load, "Load -> exp")
        wf_exp_static_13 = static_waveform(
            0, 1.3, 1000, "static")
        wf_exp_shallow_13 = transport_waveform(
            [0, 0], [1.3, 0.3], [1000, 0], n_freq_change, "shallow")
        wf_exp_static_16 = static_waveform(
            0, 1.6, 70, "exp")
        wf_exp_shallow_16 = transport_waveform(
            [0, 0], [1.6, 0.3], [70, 0], n_freq_change, "shallow")
        wf_list = [wf_load, wf_load_conveyor,
                   wf_exp_static_13, wf_exp_shallow_13,
                   wf_exp_static_16, wf_exp_shallow_16]
        
        wf_reorder = reordering_waveform(0, 1.6, 70, n_load, 0.3, 0.5, "reorder")
        wf_list.append(wf_reorder)
        
        wfs_load = WaveformSet(wf_list)
        wfs_load.write(wf_path)

    if analyse_wfms:
        
        WavPot = WavPotential( wfs_load.get_waveform(3),shim_beta = 50, shim_alpha = 10)
        return WavPot
        #pot = calculate_potentials(trap_mom ,wfs_load.get_waveform(3)) # static
        #plot_selection(pot)
        #wasd = pot.find_wells(0, mode='precise')
        #print(np.array(wasd['locs'])/um)
        #print(pot.find_wells(100, mode='precise'))

if __name__ == "__main__":
    WavPot = loading_conveyor(analyse_wfms=True)
    omegas, axes, r0, offset, V = WavPot.find_radials_3d(0)
    print(r0/um)
    WavPot.plot_radials_3d(0)