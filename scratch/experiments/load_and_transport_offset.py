#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
import copy as cp

# Loading conveyor stuff
import transport_utils as tu
import loading_utils as lu

def load_and_transport_offset(add_reordering=True, analyse_wfms=False):
    """ Generate loading waveforms, plus transporting to splitting zone with swept DC offset """
    wf_path = os.path.join(os.pardir, "waveform_files", "load_trans_offs_2Be1Ca_2017_02_20_v02.dwc.json")

    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load_and_trans = WaveformSet(waveform_file=wf_path)
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
            wfs_load_and_trans = wfs_load
        else:
            wfs_load_and_trans = WaveformSet(
                wfs_load.waveforms[:wfs_load.find_waveform("shallow", get_index=True)+1])

        default_freq = 1.1
        default_offs = 1000
        
        conveyor_offset = default_offs

        n_transport = 308
        f_well = default_freq

        splitting_loc = -422.5

        # Profiling positions
        offsets = np.linspace(-1500, 1500, 41)
        for offs in offsets:
            wf_prof = tu.transport_waveform(
                [0, splitting_loc],
                [f_well, f_well],
                [default_offs, offs],
                timesteps=n_transport,
                wfm_desc = 'trans from start -> split start, {:.1f} meV'.format(offs),
                interp_start=20, interp_end=20,
                linspace_fn=zpspace)
            wfs_load_and_trans.waveforms.append(wf_prof)

        wfs_load_and_trans.write(wf_path)

if __name__ == "__main__":
    load_and_transport_offset()
