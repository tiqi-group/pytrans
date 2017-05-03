#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

if __name__ == "__main__":
    tsteps = 11
    wf_d = WavDesiredWells(
        [np.linspace(-10, 10, tsteps)*um],
        [np.linspace(1.1, 1.3, tsteps)*MHz],
        [np.linspace(0, 1000, tsteps)*meV],
        solver_weights={'energy_threshold':200*meV,
                        'r0_u_weights':np.ones(30)*3e-2,
                        'r0_u_ss':np.ones(30)*8},
        desc="Testing")

    wf = Waveform(wf_d)
