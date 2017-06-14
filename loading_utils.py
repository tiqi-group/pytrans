#!/usr/bin/env python3
# Loading library, used to produce loading conveyor waveforms etc

import sys
sys.path.append("./")
from pytrans import *
import transport_utils as tu
import reorder as ror

def get_loading_wfms(wfm_path, force_regen_wfm=False,
                     default_freq=1.1, default_offs=1000, # used for static and return-from-shallow
                     shallow_freq=0.3, shallow_offs=-550, # used for shallow well
                     add_reordering=False,
                     add_deep_wells=False,
                     analyse_wfms=False,
                     ion_chain='2Be1Ca'):
    # If file exists already, just load it to save time
    try:
        # raise FileNotFoundError # uncomment to always regenerate file for debugging
        if force_regen_wfm:
            raise FileNotFoundError
        wfs_load = WaveformSet(waveform_file=wfm_path)
        print("Opened loading waveform ",wfm_path)
    except FileNotFoundError:
        print("Generating loading waveform ",wfm_path)
        n_load = 1001
        n_freq_change = 501

        # List of experimental-zone setting tuples
        exp_settings = [(0, default_freq, default_offs, "exp " + ion_chain)]
        # conveyor_offs = 960
        conveyor_offs = default_offs
        
        wf_load = tu.transport_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offs], n_load, "Load -> exp", linspace_fn=zpspace)
        wf_load_conveyor = tu.conveyor_rec_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offs], 2*n_load, "Load -> exp", linspace_fn=zpspace)
        st()
        wf_exp_static = tu.static_waveform(
            0, default_freq, conveyor_offs, "static")
        wf_exp_shallow = tu.transport_waveform(
            [0, 0], [default_freq, shallow_freq], [conveyor_offs, shallow_offs], n_freq_change, "shallow", linspace_fn=zpspace)
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
            plt.show()
            # analyse_wfm_radials(wf_exp_static, 0)
        
        # Default waveform, for reordering
        wf_exp_dual_species = tu.static_waveform(*exp_settings[0])
        
        # Create more deeply confining wells (maybe does not help?)
        deep_weights=dict(tu.default_weights)
        deep_weights['r0'] = 1e-3
        
        for pos, freq, offs, label in exp_settings:
            wf_exp_static = tu.static_waveform(
                pos, freq, offs, label)
            wf_exp_shallow = tu.transport_waveform(
                [pos, pos], [freq, shallow_freq], [offs, shallow_offs+1000], n_freq_change, "shallow")
            wf_exp_static_deep = tu.static_waveform(
                pos, freq, offs, label + " deep", solv_wghts=deep_weights)

            wf_list += [wf_exp_static, wf_exp_shallow]
            if add_deep_wells:
                wf_exp_static_deep = tu.static_waveform(
                    pos, freq, offs, label + " deep", solv_wghts=deep_weights)
                wf_list += [wf_exp_static_deep]

        if add_reordering:
            wf_list += ror.generate_reorder_wfms(wf_exp_dual_species,
                                                 [1.0,1.5,2.0,2.5], [0], 100)

        wfs_load = WaveformSet(wf_list)
        # st()
        wfs_load.write(wfm_path)

    if analyse_wfms:
        pot = calculate_potentials(trap_mom, wfs_load.get_waveform(2))
        plot_selection(pot)
        print(pot.find_wells(0, mode='precise'))
        print(pot.find_wells(100, mode='precise'))
    
    return wfs_load
