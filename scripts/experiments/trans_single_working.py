#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
import copy as cp

# Loading conveyor stuff
import loading_conveyor as lc
import transport_utils as tu

def trans_single():
    wf_loading_path = os.path.join(os.pardir, "waveform_files", "loading_Ca_2016_09_20_v01.dwc.json")
    wf_path = os.path.join(os.pardir, "waveform_files", "trans_single_2016_09_20_v01.dwc.json")
        
    n_transport = 500 # points
    f_transport = 1.6 # MHz
    offs_transport = 800 # mV
    split_loc = -422.5 # um
    
    # If base loading conveyor file exists already, just load it to save time
    try:
        # raise FileNotFoundError # uncomment to always regenerate loading conveyor file for debugging
        wfs_trans_single = WaveformSet(waveform_file=wf_loading_path)
        print("Loaded waveform ",wf_loading_path)
    except FileNotFoundError:
        extra_exp_settings = []
        extra_exp_settings.append( (0, f_transport, offs_transport, "exp Ca static") )
        lc.loading_conveyor(add_reordering=False,
                     add_shallow_deep=False,
                     wf_path=wf_loading_path, extra_exp_settings=extra_exp_settings)
        wfs_trans_single = WaveformSet(waveform_file=wf_loading_path)        

    print("Generating waveform ",wf_path)

    wfs_load_and_trans = WaveformSet(waveform_file=wf_loading_path)

    Ts = 200*ns
    
    # Changed lc for tu
    wf_exp_to_split = tu.transport_waveform(
        pos=[0, split_loc],
        freq=[f_transport, f_transport],
        offs=[offs_transport, offs_transport],
        timesteps=n_transport,
        wfm_desc='centre to split',
        linspace_fn=erfspace,
        Ts=Ts)

    n_shallow = 500 # points
    f_crit_splits = [0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.4,1.6]
    offs_crit_split = 800

    wfs_load_and_trans.waveforms.append(wf_exp_to_split)
    
    for f_crit in f_crit_splits:
        wf_split_to_shallow = tu.transport_waveform(
            pos=[split_loc, split_loc],
            freq=[f_transport, f_crit],
            offs=[offs_transport, offs_crit_split],
            timesteps=n_shallow,
            wfm_desc='split shallow well',
            linspace_fn=erfspace,
            Ts=Ts)

        wfs_load_and_trans.waveforms.append(wf_split_to_shallow)

    animate_trans = True
    if animate_trans:
        animate_wavpots([WavPotential(k) for k in (wf_exp_to_split, wf_split_to_shallow)], parallel=False, decimation=1)# , save_video_path='load_and_split.mp4')

    wfs_load_and_trans.write(wf_path)

if __name__ == "__main__":
    trans_single()
