#!/usr/bin/env python3

import sys, os
sys.path.append("../")
from pytrans import *
import loading_utils as lu
import transport_utils as tu
import numpy as np

if __name__ == "__main__":
    wf_name = "static_Ca_2017_12_08_v01"
    wf_path = os.path.join(os.pardir, "waveform_files", wf_name + ".dwc.json")

    print("Generating waveform ", wf_path)

    default_freq = 1.1
    default_offs = 775
            
    shallow_freq = 0.5
    shallow_offs = -300
    
    wfs_load = lu.get_loading_wfms(os.path.join(os.pardir, "waveform_files", "loading_Ca_2017_12_08_v01.dwc.json"),
                                   default_freq=default_freq, default_offs=default_offs,
                                   shallow_freq=shallow_freq, shallow_offs=shallow_offs,
                                   add_reordering=False, ion_chain='Ca', force_regen_wfm=False)

    freqs = np.linspace(1.4, 2.2, 5)
    offsets = np.arange(-400, 1800, 100)    
    
    for freq in freqs:
        for offs in offsets:
            wfs_load.waveforms.append(tu.static_waveform(global_exp_pos, freq, offs, "exp Ca static"))

    wfs_load.write(wf_path, fix_voltage_limits=True)
