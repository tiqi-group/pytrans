#!/usr/bin/env python3
# Check/debug a few numeric issues with different solvers; for example MOSEK tends to cause 'wiggles' where there should be none

import sys
sys.path.append("../")
from pytrans import *
import transport_utils as tu

def numeric_test():
    # Set up problem
    td_weights = True

    tsteps = 100
    pos = np.linspace(0, 0, tsteps)
    freq = np.linspace(1.1, 1.1, tsteps)
    offs = np.linspace(1000, 1000, tsteps)
    if td_weights:
        solver_weights={'r0':5e-10,
                        'r0_u_weights':np.ones(30), # all electrodes uniform
                        'r0_u_ss':np.ones(30)*default_elec_voltage,
                        'r1':5e-4,'r2':2e-6}
    else:
        solver_weights={'r0':5e-10,
                        'r0_u_weights':np.ones(30), # all electrodes uniform
                        'r0_u_ss':np.full(30, default_elec_voltage),
                        'r1':0,'r2':0}

    wf_d = WavDesiredWells([pos*um], [freq*MHz], [offs*meV],
                           solver_weights=solver_weights,
                           desired_potential_params={'energy_threshold':2*meV},
                           force_static_ends=True,
                           Ts=200*ns,
                           desc="Test")
    wf = Waveform(wf_d, solver='ECOS', max_iters=1000, abstol=1e-16, reltol=1e-15, feastol=1e-8)
    # wf = Waveform(wf_d)


    print(wf.samples[:,[0,-1]])

    wfp = WavPotential(wf)    
    wfp.plot_voltages()
    wfp.plot()

    # wf2 = tu.transport_waveform(pos, freq, offs, tsteps, "")
    # wfp2 = WavPotential(wf2)
    # wfp2.plot_voltages()

    plt.show()

def compare_static_shallow():
    freq = 1.1
    offs = 1000
    shallow_freq = 0.5
    shallow_offs = -300

    wf_static = tu.static_waveform(0, freq, offs, "static")
    wf_shallow = tu.shallow_waveform([freq, shallow_freq], [offs, shallow_offs], 100)

if __name__ == "__main__":
    compare_static_shallow()
    # numeric_test()
    # transport_waveform_test()
