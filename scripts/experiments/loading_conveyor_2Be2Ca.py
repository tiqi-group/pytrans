#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
from loading_conveyor import local_weights, static_waveform, transport_waveform, transport_waveform_multiple, conveyor_waveform
    
def loading_conveyor_2Be2Ca(add_reordering=True, analyse_wfms=False):
    wf_path = os.path.join(os.pardir, "waveform_files", "loading_2Be2Ca_2016_07_11_v07.dwc.json")

    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ",wf_path)
        loading_pos = -1870 + 1
        loading_freq = 0.7
        loading_offs = 600
        n_load = 1001
        n_short_transport = 501
        n_freq_change = 200
        default_freq = 0.9
        default_offs = +800

        # List of experimental-zone setting tuples
        exp_settings = [(0, default_freq, default_offs, "exp BeCa")]
        # conveyor_end_freq = loading_freq
        conveyor_end_freq = default_freq
        conveyor_end_offs = default_offs
        low_freq_freq = 0.2
        low_freq_offset = 500
        
        wf_load = transport_waveform(
            [loading_pos, 0], [loading_freq, conveyor_end_freq],
            [loading_offs, conveyor_end_offs], n_load, "Load -> exp")
        wf_load_conveyor = conveyor_waveform(
            [loading_pos, 0], [loading_freq, conveyor_end_freq],
            [loading_offs, conveyor_end_offs], n_load, "Load -> exp")

        # Convert from final conveyor (somewhat shallow) to default potential
        wf_exp_conveyor_to_default = transport_waveform(
            [0, 0], [conveyor_end_freq, default_freq],
            [conveyor_end_offs, default_offs], n_freq_change, "exp conveyor -> static")

        wf_exp_default_static = static_waveform(
            0, default_freq, default_offs, "static")

        # Lower trapping potential
        wf_exp_shallow = transport_waveform(
            # offset of -300 determined from simulation
            [0, 0], [default_freq, low_freq_freq], [default_offs, -300], n_freq_change, "shallow") 
        wf_list = [wf_load, wf_load_conveyor,
                   wf_exp_conveyor_to_default,
                   wf_exp_default_static,
                   wf_exp_shallow]

        add_shallow_transport = False
        if add_shallow_transport:
            shallow_med_freq = 0.75
            shallow_med_offs = 900
            wf_load_shallow_med = transport_waveform(
                [loading_pos, 0], [loading_freq, shallow_med_freq], [loading_offs, shallow_med_offs],
                n_load, "Load -> exp shallow medium")
            wf_exp_shallow_med = transport_waveform(
                [0, 0], [default_freq, shallow_med_freq], [default_offs, shallow_med_offs],
                n_freq_change, "shallow medium")
            wf_list += [wf_load_shallow_med, wf_exp_shallow_med]
        
        add_exp_settings = False        
        wf_exp_dual_species = static_waveform(*exp_settings[0])
        if add_exp_settings:
            # Default waveform, for reordering

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

        # Generate waveforms for transporting the central well to a
        # side position, for viewing on the Be camera
        add_shifted_pos_load = True
        add_shifted_pos_exp = True
        shifted_pos_load = [1600] #np.linspace(1450, 1650, 5) # relative to loading zone
        shifted_pos_exp = [-270] #np.linspace(-270, -300, 4) # relative to centre
        if add_shifted_pos_load:
            for new_pos in shifted_pos_load:
                wf_shifted_load = transport_waveform(
                    [loading_pos, loading_pos+new_pos],
                    [loading_freq, loading_freq],
                    [loading_offs, loading_offs],
                    n_short_transport,
                    "shifted pos load + "+str(new_pos)+" um")
                wf_list += [wf_shifted_load]
        
        if add_shifted_pos_exp:
            for new_pos in shifted_pos_exp:
                wf_shifted_exp = transport_waveform(
                    [0, new_pos],
                    [default_freq, default_freq],
                    [default_offs, default_offs],
                    n_short_transport,
                    "shifted pos exp "+str(new_pos)+" um")
                wf_list += [wf_shifted_exp]

        if add_reordering:
            wf_list += generate_reorder_2Be2Ca_wfms(default_freq, default_offs,
                                             [2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
                                             [default_offs]*6,
                                                    [500]*6)
            wf_list += generate_reorder_wfms(wf_exp_dual_species,
                                             [0.4,0.5,0.6],
                                             [0.6,0.7,0.8,1.0],
                                             100)
        
        wfs_load = WaveformSet(wf_list)
        wfs_load.write(wf_path)

    if analyse_wfms:
        pot = WavPotential(wfs_load.get_waveform(2))
        pot.plot_one_wfm(0)
        print(pot.find_wells(0, mode='precise'))
        plt.show()

if __name__ == "__main__":
    loading_conveyor_2Be2Ca(analyse_wfms=False)
