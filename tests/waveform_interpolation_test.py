#!/usr/bin/env python3
# Interpolates between two waveforms, with different weighting factors

import sys
sys.path.append("../")
from pytrans import *

def add_interp_wfms(wfm_file, wfm1, wfm2, new_wfms,
                    wfm_string=lambda frac: "profiling"):
    """ wfm_file: waveform file to extend 
    wfm1: string unique to the first waveform
    wfm2: string unique to 2nd waveform
    new_wfms: how many new waveforms to interpolate; add 2 for start/end waveforms (which should have same samples as wfm1 and wfm2) since linspace fn is used
    wfm_string: function that is input a number between 0 and 1 (using the linspace values produced based on new_wfms), and returns the string to name each successive waveform
    """
    wfs = WaveformSet(waveform_file=wfm_file)
    wf1_samples = wfs.find_waveform(wfm1).samples
    wf2_samples = wfs.find_waveform(wfm2).samples

    new_wfm_factors = np.linspace(0, 1, new_wfms)
    for fact in new_wfm_factors:
        new_samples = (1-fact) * wf1_samples + fact * wf2_samples # linear interpolation
        new_wfm = Waveform(wfm_string(fact), 0, "", new_samples)
        new_wfm.set_new_uid()
        wfs.waveforms.append(new_wfm)
    
    return wfs

if __name__ == "__main__":
    wfs = add_interp_wfms("../waveform_files/load_split_2Be1Ca_2017_02_21_v04_mod_v03.dwc.json",
                          "to -6.0 um",
                          "to -4.0 um",
                          11,
                          lambda fact: "interp profiling to {:.1f} um".format( (1-fact)*(-6) + fact*(-4) ))

    wfs.write("../waveform_files/load_split_2Be1Ca_2017_02_21_v04_mod_v04.dwc.json")
