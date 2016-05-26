#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

def single_waveform():
    wf_path = os.path.join(os.pardir, "waveform_files", "single_test_waveform.dwc.json")
    w_desired = WavDesiredWells(np.array([0])*um,
                                np.array([1.5])*MHz,
                                np.array([800])*meV,
                                solver_weights={'energy_threshold':200*meV,
                                                'r0_u_weights':np.ones(30)*3e-2,
                                                'r0_u_ss':np.ones(30)*8},
                                desc="Single waveform test")
    wfs = WaveformSet([Waveform(w_desired)])
    wfs.get_waveform(1).samples[10,0] = 0
    wfs.write(wf_path)

def analyze_waveform():
    wf_path = os.path.join(os.pardir, "waveform_files", "single_test_waveform.dwc.json")

    pot = calculate_potentials(trap_mom, WaveformSet(waveform_file=wf_path).get_waveform(1))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    pot.plot_one_wfm(0, ax)
    plt.show()

if __name__ == "__main__":
    # load_to_exp()
    single_waveform()
    analyze_waveform()

    
    
