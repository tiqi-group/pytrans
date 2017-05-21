#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
import transport_utils as tu

def simple_waveform_test():
    tsteps = 101
    wf_d = WavDesiredWells(
        [np.linspace(-500, 500, tsteps)*um],
        [np.linspace(0.9, 1.5, tsteps)*MHz],
        [np.linspace(-500, 500, tsteps)*meV],
        solver_weights={'energy_threshold':200*meV,
                        'r0_u_weights':np.ones(30)*3e-2,
                        'r0_u_ss':np.ones(30)*8},
        force_static_ends=False,
        Ts=100*ns,
        desc="Testing")
    
    wf_d_stat = WavDesiredWells( # static ends
        [np.linspace(-500, 500, tsteps)*um],
        [np.linspace(0.9, 1.5, tsteps)*MHz],
        [np.linspace(-500, 500, tsteps)*meV],
        solver_weights={'energy_threshold':200*meV,
                        'r0_u_weights':np.ones(30)*3e-2,
                        'r0_u_ss':np.ones(30)*8},
        force_static_ends=True,
        Ts=100*ns,
        desc="Testing, static ends")

    wf = Waveform(wf_d)
    wf_stat = Waveform(wf_d_stat)

    wfp = WavPotential(wf)
    wfp_stat = WavPotential(wf_stat)

    wfp.plot_voltages()
    wfp_stat.plot_voltages()
    # st()
    
    # wfp.animate_wfm(decimation=1, wdp=wf_d)
    wfp_stat.animate_wfm(decimation=1, wdp=wf_d_stat)

def transport_waveform_multiple_test():
    tsteps = 501
    fw = 1.1
    wf = tu.transport_waveform_multiple(
        [[-800,0],[0,800]],
        [[fw,fw],[fw,fw]],
        [[1000,1000],[1000,1000]],
        tsteps,
        linspace_fn=zpspace,
        wfm_desc="Trans wf multiple test")
    wfp = WavPotential(wf)
    wfp.animate_wfm(decimation=1)

if __name__ == "__main__":
    transport_waveform_multiple_test()
