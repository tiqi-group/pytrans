#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

local_weights = {'r0_u_weights':np.ones(30)*1e-9,
                 'r0_u_ss':np.ones(30)*0}

linspace_fn = np.linspace

def plotting_test():
    #wf_path = os.path.join(os.pardir, "waveform_files", "loading_conveyor_2016_06_02_v01.dwc.json")
    wf_path_matlab = os.path.join(os.pardir, "waveform_files", "loading_conveyor_Ts620_2016_04_29_v01.dwc.json")

    # Copied and pasted from experiments/loading_conveyor.py
    wdw = WavDesiredWells(
            [linspace_fn(-1870, 0, 101)*um],
            [linspace_fn(1.1, 1.3, 101)*MHz],
            [linspace_fn(600, 1000, 101)*meV],
            solver_weights=local_weights,
            desc="asdf")
    
    # wf_test = Waveform(wdw)
    wf_test = Waveform(wdw)
    wf_test_matlab = WaveformSet(waveform_file=wf_path_matlab).get_waveform(1)
    pot = calculate_potentials(trap_mom, wf_test)
    pot_matlab = calculate_potentials(trap_mom, wf_test_matlab)
    ax = wdw.plot(0, trap_mom.transport_axis)
    pot.plot_range_of_wfms([0], ax)
    pot_matlab.plot_range_of_wfms([0], ax)
    plt.show()
    
    well_data = pot.find_wells(0)
    locs = trap_mom.transport_axis[well_data['min_indices']]/um
    print(locs, well_data['freqs'])

if __name__ == "__main__":
    plotting_test()
