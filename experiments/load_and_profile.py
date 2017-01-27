#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
import copy as cp

# Loading conveyor stuff
import transport_utils as tu
import loading_utils as lu

def load_and_profile(add_reordering=True, analyse_wfms=False):
    """ Generate loading/profile waveforms, with swept offset """
    wf_path = os.path.join(os.pardir, "waveform_files", "load_profile_2Be1Ca_2017_01_26_v01.dwc.json")

    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load_and_profile = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ", wf_path)
        # use existing loading conveyor file to save time - need to regenerate if not available
        wfs_load = lu.get_loading_wfms("loading_2Be1Ca_2017_01_25_v01.dwc.json",
                                       add_reordering=True, ion_chain='2Be1Ca')
        
        # truncate waveforms after the first shallow one
        reordering = True
        num_reorder_wfms = 4 if reordering else 0
        if reordering:
            wfs_load_and_profile = wfs_load
        else:
            wfs_load_and_profile = WaveformSet(
                wfs_load.waveforms[:wfs_load.find_waveform("shallow", get_index=True)+1])

        default_freq = 1.1
        default_offs = 1000
        
        conveyor_offset = default_offs

        n_transport = 308
        f_well = default_freq

        # Profiling positions
        profile_pos = np.linspace(-50, 50, 51)
        for pp in profile_pos:
            wf_prof = tu.transport_waveform(
                [0, pp],
                [f_well, f_well],
                [default_offs, default_offs],
                timesteps=100,
                wfm_desc="exp->profiling, to {a} um".format(a=pp),
                interp_start=20, interp_end=20)
            wfs_load_and_profile.waveforms.append(wf_prof)

        wfs_load_and_profile.write(wf_path)

if __name__ == "__main__":
    load_and_profile()
