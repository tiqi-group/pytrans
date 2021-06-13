#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
import copy as cp

# Loading conveyor stuff
import transport_utils as tu

# Splitting (need to refactor soon - there's a lot of unneeded stuff in splitting.py!)
import splitting as sp

def trans_load_to_split(add_reordering=True, analyse_wfms=False):
    wf_path = os.path.join(os.pardir, "waveform_files", "split_swept_2Be_2016_10_25_v01.dwc.json")
    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load_and_split = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ",wf_path)
        # use existing loading conveyor file to save time - need to regenerate if not available
        wf_load_path = os.path.join(os.pardir, "waveform_files", "loading_2016_06_21_v01.dwc.json")
        wfs_load = WaveformSet(waveform_file=wf_load_path)

        # truncate waveforms from the first shallow one
        wfs_load_and_split = WaveformSet(wfs_load.waveforms[:wfs_load.find_waveform("shallow", get_index=True)+1])

        n_transport = 1000
        load_to_split, wf_split, wf_split_swept = sp.split_waveforms_many_resamples(
            0, 1.1, 960,
            [-844, 0], [1.3,1.3], [960, 960],
            -422.5, 1.1,
            n_transport=n_transport,
            electrode_subset=[3,4,5,6,7,18,19,20,21,22]) # left splitting group
        
        wfs_load_and_split.waveforms.append(load_to_split)
        wfs_load_and_split.waveforms.append(wf_split)
        wf_far_to_exp = tu.transport_waveform_multiple(
            [[-844,0],[0,600]],
            [[1.3,1.3],[1.3,1.3]],
            [[960,960],[960,960]],
            2*n_transport,
            "-far to centre, centre to +far")
        wfs_load_and_split.waveforms.append(wf_far_to_exp)

        # Append extra splitting waveforms
        wf_list = list(wf_split_swept)
        wfs_swept = WaveformSet(wf_list)
        write_only_swept = False
        if write_only_swept:
            wfs_swept.write(wf_path)
        else:
            wfs_load_and_split.waveforms += wf_list
            wfs_load_and_split.write(wf_path)

    # Create a single unified testing waveform, made up of the individual transports
    add_unified_waveform = False
    if add_unified_waveform:
        test_waveform_present = wfs_load_and_split.get_waveform(-1).desc == "trans + split, then reverse"
        if test_waveform_present:
            exp_to_split_wfm = wfs_load_and_split.get_waveform(-4)
            split_wfm = wfs_load_and_split.get_waveform(-3)
            far_to_cent_wfm = wfs_load_and_split.get_waveform(-2)
        else:
            exp_to_split_wfm = wfs_load_and_split.get_waveform(-3)
            split_wfm = wfs_load_and_split.get_waveform(-2)
            far_to_cent_wfm = wfs_load_and_split.get_waveform(-1)            

        #trans_split_forward = np.hstack([exp_to_split_wfm.samples, split_wfm.samples[:,:-500]])
        trans_split_forward = np.hstack([exp_to_split_wfm.samples, split_wfm.samples, far_to_cent_wfm.samples])
        trans_split_for_rev = np.hstack([trans_split_forward, np.fliplr(trans_split_forward)])
        wf_trans_split_for_rev = Waveform("trans + split, then reverse", 0, "", trans_split_for_rev)
        wf_trans_split_for_rev.set_new_uid()

        if test_waveform_present:
            wfs_load_and_split.waveforms[-1] = wf_trans_split_for_rev
        else:
            wfs_load_and_split.waveforms.append(wf_trans_split_for_rev)

        wf_forward = Waveform("", 0, "", trans_split_forward)
        pot_forward = WavPotential(wf_forward)
        pot_for_rev = WavPotential(wf_trans_split_for_rev)

        print(pot_forward.find_wells(-1))
        pot_for_rev.animate_wfm(decimation=1)
        wfs_load_and_split.write(wf_path)

    # Manually alter splitting offset by changing voltage [obsolete]
    alter_splitting_offset = False
    if alter_splitting_offset:
        wf_dbg_path = os.path.join(os.pardir, "waveform_files", "load_split_debug_2016_06_22_v01.dwc.json")
        wf_exp_to_split = wfs_load_and_split.find_waveform("trans from start -> split start")
        wf_split = wfs_load_and_split.find_waveform("split apart")
        wf_for_rev = wfs_load_and_split.find_waveform("then reverse")
        wf_split.samples[physical_electrode_transform[[2, 17]], :] -= 0.8 # decrease voltage
        wf_for_rev.samples = merge_waveforms_for_rev([wf_exp_to_split, wf_split]).samples

        # check the behaviour
        pot_for_rev = WavPotential(wf_for_rev)
        pot_for_rev.animate_wfm()
        wfs_load_and_split.write(wf_dbg_path)

if __name__ == "__main__":
    trans_load_to_split()
 
