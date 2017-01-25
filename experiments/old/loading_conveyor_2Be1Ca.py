#!/usr/bin/env python3

### DO NOT USE THIS FILE, IT IS OBSOLETE ###
### USE loading_utils.py ###

import sys
sys.path.append("../")
from pytrans import *
import reorder as ror
import transport_utils as tu

local_weights = {'r0':1e-5,
                 'r0_u_weights':np.ones(30), # all electrodes uniform
                 'r0_u_ss':np.ones(30)*8,
                 'r1':1e-6,'r2':1e-7}

local_potential_params={'energy_threshold':10*meV}

def plot_selection(pot):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    legends = []
    #pot.plot_range_of_wfms(np.linspace(400,800,100, dtype='int'))
    pot.plot_range_of_wfms(100, ax)
    # for k in range(0,201,10):        
    #     pot.plot_one_wfm(k, ax)
    #     legends.append(str(k))

    plt.legend(legends)
    plt.show()

def analyse_wfm_radials(wfm, wfm_idx=0, plot_radials=True):
    wavpot = WavPotential(wfm, shim_beta=0, shim_alpha=0) #, rf_v=)
    omegas, axes, r0, offset, V = wavpot.find_radials_3d(wfm_idx)
    if plot_radials:
        wavpot.plot_radials(wfm_idx, mode='3d')
    # 2.76, 3.27 radials for 760meV offset
    # 2.76, 3.35 radials for 960meV offset
    return omegas, axes, r0, offset, V
    
def loading_conveyor_2Be1Ca(add_reordering=True, analyse_wfms=False):
    wf_path = os.path.join(os.pardir, "waveform_files", "loading_2Be1Ca_2016_12_07_v01.dwc.json")

    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ",wf_path)
        n_load = 1001
        n_freq_change = 200
        default_freq = 1.1
        default_offs = 1000
        # default_offs = 760

        shallow_freq = 0.3

        # List of experimental-zone setting tuples
        exp_settings = [(0, default_freq, default_offs, "exp 2Be1Ca")]
        # conveyor_offset = 960
        conveyor_offset = default_offs
        shallow_offset = -550
        
        wf_load = tu.transport_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offset], n_load, "Load -> exp")
        wf_load_conveyor = tu.conveyor_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offset], n_load, "Load -> exp")
        wf_exp_static = tu.static_waveform(
            0, default_freq, conveyor_offset, "static")        
        wf_exp_shallow = tu.transport_waveform(
            [0, 0], [default_freq, shallow_freq], [conveyor_offset, shallow_offset], n_freq_change, "shallow")
        wf_list = [wf_load, wf_load_conveyor,
                   wf_exp_static, wf_exp_shallow]

        analyse_static_radials = False
        if analyse_static_radials:
            n_timesteps = wf_exp_shallow.samples.shape[1]
            vec_mode1 = np.zeros((n_timesteps,2))
            vec_mode2 = np.zeros_like(vec_mode1)
            
            for k in range(n_timesteps):
                omegas, axes, r0, offset, V = analyse_wfm_radials(wf_exp_shallow, -1, False)
                vec_mode1[k, 0:2] = omegas[0]*axes[1,1:3]
                vec_mode2[k, 0:2] = omegas[1]*axes[2,1:3]

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            plt.plot(np.arange(n_timesteps), vec_mode1[:,0], vec_mode1[:,1], 'r')
            plt.plot(np.arange(n_timesteps), vec_mode2[:,0], vec_mode2[:,1], 'g')
            # st()
            plt.show()
            # analyse_wfm_radials(wf_exp_static, 0)
        
        # Default waveform, for reordering
        wf_exp_dual_species = tu.static_waveform(*exp_settings[0])
        
        # Create more deeply confining wells (maybe does not help?)
        deep_weights=dict(local_weights)
        deep_weights['r0'] = 1e-3        
        for pos, freq, offs, label in exp_settings:
            wf_exp_static = tu.static_waveform(
                pos, freq, offs, label)        
            wf_exp_shallow = tu.transport_waveform(
                [pos, pos], [freq, 0.3], [offs, 0], n_freq_change, "shallow")
            wf_exp_static_deep = tu.static_waveform(
                pos, freq, offs, label + " deep", solv_wghts=deep_weights)

            wf_list += [wf_exp_static, wf_exp_shallow, wf_exp_static_deep]

        if add_reordering:
            wf_list += ror.generate_reorder_wfms(wf_exp_dual_species,
                                                 [1.0,1.5,2.0,2.5],
                                                 [0],
                                                 100)
        
        wfs_load = WaveformSet(wf_list)
        wfs_load.write(wf_path)

    if analyse_wfms:
        pot = calculate_potentials(trap_mom, wfs_load.get_waveform(2))
        plot_selection(pot)
        print(pot.find_wells(0, mode='precise'))
        print(pot.find_wells(100, mode='precise'))

if __name__ == "__main__":
    loading_conveyor_2Be1Ca(analyse_wfms=False)
