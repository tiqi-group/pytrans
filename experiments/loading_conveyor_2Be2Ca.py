#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
from loading_conveyor import local_weights, static_waveform, transport_waveform, transport_waveform_multiple, conveyor_waveform
    
def loading_conveyor_2Be2Ca(add_reordering=True, analyse_wfms=False):
    wf_path = os.path.join(os.pardir, "waveform_files", "loading_2Be2Ca_2016_07_07_v01.dwc.json")

    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ",wf_path)
        n_load = 1001
        n_freq_change = 200
        default_freq = 1.0
        default_offs = +700

        # List of experimental-zone setting tuples
        exp_settings = [(0, default_freq, default_offs, "exp BeCa")]
        conveyor_offset = default_offs
        low_freq_freq = 0.2
        low_freq_offset = +500
        
        wf_load = transport_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offset], n_load, "Load -> exp")
        wf_load_conveyor = conveyor_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offset], n_load, "Load -> exp")
        wf_exp_static_13 = static_waveform(
            0, default_freq, conveyor_offset, "static")
        wf_exp_shallow_13 = transport_waveform(
            # offset of -300 determined from simulation
            [0, 0], [default_freq, low_freq_freq], [conveyor_offset, -300], n_freq_change, "shallow") 
        wf_list = [wf_load, wf_load_conveyor,
                   wf_exp_static_13, wf_exp_shallow_13]

        # Default waveform, for reordering
        wf_exp_dual_species = static_waveform(*exp_settings[0])
        
        # Create more deeply confining wells (maybe does not help?)
        deep_weights=dict(local_weights)
        deep_weights['r0'] = 1e-3        
        for pos, freq, offs, label in exp_settings:
            wf_exp_static = static_waveform(
                pos, freq, offs, label)        
            wf_exp_shallow = transport_waveform(
                [pos, pos], [freq, low_freq_freq], [offs, low_freq_offset], n_freq_change, "shallow")
            wf_exp_static_deep = static_waveform(
                pos, freq, offs, label + " deep", solv_wghts=deep_weights)

            wf_list += [wf_exp_static, wf_exp_shallow, wf_exp_static_deep]

        if add_reordering:
            wf_list += generate_reorder_2Be2Ca_wfms(default_freq, default_offs,
                                             [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
                                             [default_offs]*6,
                                                    [500]*6)
            wf_list += generate_reorder_wfms(wf_exp_dual_species,
                                             [0.4,0.5,0.6],
                                             [0.6,0.7,0.8,1.0],
                                             100)
        
        wfs_load = WaveformSet(wf_list)
        wfs_load.write(wf_path)

    if analyse_wfms:
        pot = calculate_potentials(trap_mom, wfs_load.get_waveform(2))
        plot_selection(pot)
        print(pot.find_wells(0, mode='precise'))
        print(pot.find_wells(100, mode='precise'))

if __name__ == "__main__":
    loading_conveyor_2Be2Ca(analyse_wfms=True)
