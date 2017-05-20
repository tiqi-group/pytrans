#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

if __name__ == "__main__":
    tsteps = 101
    wf_d = WavDesiredWells(
        [np.linspace(-500, 500, tsteps)*um],
        [np.linspace(1.1, 1.3, tsteps)*MHz],
        [np.linspace(-500, 500, tsteps)*meV],
        solver_weights={'energy_threshold':200*meV,
                        'r0_u_weights':np.ones(30)*3e-2,
                        'r0_u_ss':np.ones(30)*8},
        force_static_ends=False,
        desc="Testing")
    
    wf_d_stat = WavDesiredWells( # static ends
        [np.linspace(-500, 500, tsteps)*um],
        [np.linspace(1.1, 1.3, tsteps)*MHz],
        [np.linspace(-500, 500, tsteps)*meV],
        solver_weights={'energy_threshold':200*meV,
                        'r0_u_weights':np.ones(30)*3e-2,
                        'r0_u_ss':np.ones(30)*8},
        force_static_ends=True,
        desc="Testing, static ends")

    wf = Waveform(wf_d)
    wf_stat = Waveform(wf_d_stat)

    wfp = WavPotential(wf)
    wfp_stat = WavPotential(wf_stat)

    wfp_stat.plot_voltages()
    # st()
    
    # wfp.animate_wfm(decimation=1, wdp=wf_d)
    wfp_stat.animate_wfm(decimation=1, wdp=wf_d_stat)
