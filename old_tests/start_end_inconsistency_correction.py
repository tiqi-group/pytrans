#!/usr/bin/env python3
# Makes sure that for a standard 2Be1Ca waveform, the reorder waveforms all begin/end at exactly the same locations as the static well.

import sys
sys.path.append("../")
from pytrans import *

def correct_reorder_wfms(wfm_file, reorder_idces=[6,7,8,9,10]):
    wfs = WaveformSet(waveform_file=wfm_file)
    wf_static_samples = wfs.find_waveform("static").samples

    for idx in reorder_idces:
        wf_reorder_samples = wfs.get_waveform(idx).samples # zero-indexed; subtract 1 from what the waveform numbers shown in Ionizer
        wf_diff = wf_static_samples - wf_reorder_samples[:,[0]]
        if np.abs(wf_diff.sum()) > 5:
            warnings.warn("Large difference seen between reorder and static waveforms; should be a small difference."
                          "Please make sure the right operation is being run! Not applying for Wfm {:d}".format(idx))
        elif np.abs(wf_diff.sum()) < 0.0001:
            warnings.warn("Very small difference between reorder and static waveforms; is the reorder already corrected? Not applying for Wfm {:d}".format(idx))
        wf_reorder_samples += np.tile(wf_diff, [1, wf_reorder_samples.shape[1]])

    return wfs

if __name__ == "__main__":
    wfs = correct_reorder_wfms("../waveform_files/load_split_2Be1Ca_2017_02_21_v04_mod_v05.dwc.json")
    wfs.write("../waveform_files/load_split_2Be1Ca_2017_02_21_v04_mod_v06.dwc.json")
