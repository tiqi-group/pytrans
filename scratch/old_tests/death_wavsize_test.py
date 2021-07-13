#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

if __name__ == "__main__":
    wf_list = []
    num_waveforms = 100
    for k in range(num_waveforms):
        data = np.ones([32, 17000//num_waveforms])
        wf = Waveform("big!"+str(k),0,"", data)
        wf_list.append(wf)
        
    wfs = WaveformSet(wf_list)
    wfs.write(os.path.join(os.pardir, "waveform_files", "test_size.dwc.json"))
