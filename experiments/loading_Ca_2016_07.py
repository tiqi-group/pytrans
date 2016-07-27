#!/usr/bin/env python3

import sys
sys.path.append("../")
from loading_conveyor import *

if __name__ == "__main__":
    extra_exp_settings = []

    freqs = np.linspace(1.4, 2.2, 5)
    offsets = np.arange(-400, 1800, 100)
    
    for freq in freqs:
        for offs in offsets:
            extra_exp_settings.append( (0, freq, offs, "exp Ca static") )

    wf_path = os.path.join(os.pardir, "waveform_files", "loading_Ca_2016_07_26_v01.dwc.json")

    loading_conveyor(add_reordering=False,
                     add_shallow_deep=False,
                     wf_path=wf_path, extra_exp_settings=extra_exp_settings)
