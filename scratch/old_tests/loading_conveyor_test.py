#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

def plot_selection(pot):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    legends = []
    for k in range(0,101,20):        
        pot.plot_one_wfm(k, ax)
        legends.append(str(k))

    plt.legend(legends)
    plt.show()

def load_to_exp():
    wf_path = os.path.join(os.pardir, "waveform_files", "load_to_exp_2016_05_24_v02.dwc.json")

    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # always generate for debugging
        wfs_load = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ",wf_path)
        n_load = 101
        # Desired constraints
        w_load = WavDesiredWells(
            [np.linspace(-1870, 0, n_load)*um],
            [np.linspace(1.1, 1.3, n_load)*MHz],
            [np.linspace(600, 1000, n_load)*meV],
            solver_weights={'energy_threshold':200*meV,
             'r0_u_weights':np.ones(30)*3e-2,
             'r0_u_ss':np.ones(30)*8},
            desc="Load -> exp zone, 1.1 -> 1.3 MHz, 0.6 -> 1.0 eV")
        # Create a waveform set from a waveform
        wfs_load = WaveformSet([Waveform(w_load)])
        wfs_load.write(wf_path)

    pot = calculate_potentials(trap_mom, wfs_load.get_waveform(1))
    plot_selection(pot)

def loading_transport_test():
    
    
if __name__ == "__main__":
    # load_to_exp()

    loading_transport_test()
