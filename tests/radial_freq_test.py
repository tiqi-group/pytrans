##!/usr/bin/env python3
import sys
sys.path.append("../")
sys.path.append("../experiments")
from pytrans import *
import transport_utils as tu
import analysis_utils as au
import scipy.optimize as sopt

if __name__ == "__main__":
    """ Calculate the radial frequencies for a waveform, given a range of RF voltages """
    wfs = WaveformSet(waveform_file="../waveform_files/loading_Ca_2016_07_26_v01.dwc.json")
    wf = wfs.get_waveform(63)

    rfs = np.linspace(370, 390, 100)
    freqs, offsets = au.find_radials(wf, rfs, plot_results=True)

    plt.show()
