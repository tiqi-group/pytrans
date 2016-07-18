#!/usr/bin/env python3
# Test of radial field calculation (VN: just a testbed for my learning)
import sys
sys.path.append("../")
sys.path.append("../experiments")
from pytrans import *
from loading_conveyor import static_waveform, transport_waveform

def radial_field_test():
    wf = Waveform("", 0, "", np.zeros((32,1)))
    wp = WavPotential(wf, rf_v=0, shim_alpha=-0.964, shim_beta=0.459)
    # wp = WavPotential(wf, rf_v=0)
    wpot = wp.add_potentials(0, slice_ind=trap_mom.pot3d.ntot//2)
    st()
    plt.plot(wpot, ':')
    # plt.plot(trap_mom.transport_axis, wpot)

if __name__ == "__main__":
    radial_field_test()
    plt.show()
