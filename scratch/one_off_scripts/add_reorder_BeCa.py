#!/usr/bin/env python3
# Adds reordering waveforms in BeCa-style to a given waveform + index

import sys
sys.path.append("../")
from pytrans import *
import reorder as ro

#push: 0.2 -> 0.8, 0.1 steps; using 0.4
#twist: 0.3 -> 2, 0.1 steps; using 0.7

if __name__ == "__main__":
    wf_f = "../waveform_files/load_split_2Be1Ca_2017_02_21_v04.dwc.json"
    wfs = WaveformSet(waveform_file=wf_f)

    old_reorder = wfs.find_waveforms('reorder')
    for oreo in old_reorder:
        wfs.waveforms.remove(oreo)

    wfm_static = wfs.find_waveforms("static")[0]

    new_reorder = ro.generate_reorder_wfms(wfm_static,
                                        push_v_vec=[0.2, 0.4, 0.6, 0.8, 1.0],
                                        twist_v_vec=[0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
                                        timesteps=100)

    # UNFINISHED; CONTINUE HERE
    st()
