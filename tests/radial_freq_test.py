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

    single_Ca = False
    three_Be = True
    
    if single_Ca:
        wfs = WaveformSet(waveform_file="../waveform_files/loading_Ca_2016_07_26_v01.dwc.json")
        wf = wfs.get_waveform(63)

        rfs = np.linspace(370, 390, 100)
        freqs, offsets = au.find_radials(wf, rfs, plot_results=True)

        plt.show()

    if three_Be:
        wfs = WaveformSet(waveform_file="../waveform_files/load_split_2Be1Ca_2017_02_22_v02.dwc.json")
        # wf = WavPotential(wfs.find_waveform("0.00"))
        wf = WavPotential(wfs.find_waveform("static"))
        wf.plot_radials(0, mode='3d')
        plt.show()
