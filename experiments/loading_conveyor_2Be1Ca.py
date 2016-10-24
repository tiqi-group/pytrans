#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
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
    
def loading_conveyor(add_reordering=True, analyse_wfms=False):
    wf_path = os.path.join(os.pardir, "waveform_files", "loading_2Be1Ca_2016_07_11_v02.dwc.json")

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
        default_offs = 960

        # List of experimental-zone setting tuples
        exp_settings = [(0, default_freq, default_offs, "exp 2Be1Ca")]
        conveyor_offset = 960
        
        wf_load = tu.transport_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offset], n_load, "Load -> exp")
        wf_load_conveyor = tu.conveyor_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offset], n_load, "Load -> exp")
        wf_exp_static_13 = tu.static_waveform(
            0, default_freq, conveyor_offset, "static")
        wf_exp_shallow_13 = tu.transport_waveform(
            [0, 0], [default_freq, 0.3], [conveyor_offset, 0], n_freq_change, "shallow")
        wf_list = [wf_load, wf_load_conveyor,
                   wf_exp_static_13, wf_exp_shallow_13]

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
            wf_list += generate_reorder_wfms(wf_exp_dual_species,
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
    loading_conveyor(analyse_wfms=True)
