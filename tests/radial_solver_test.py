#!/usr/bin/env python3
import sys
sys.path.append("../")
sys.path.append("../experiments")
from pytrans import *
import transport_utils as tu
# from loading_conveyor import static_waveform, transport_waveform
import scipy.optimize as sopt

def radial_scan(freqs, offsets):
    fig = plt.figure(figsize=(25,25), dpi=30)    
    
    for f, freq in enumerate(freqs):
        for o, offs in enumerate(offsets):
            sp_ax = fig.add_subplot(5, 5, len(freqs)*f+o+1)
            wf = tu.static_waveform(0, freq, offs, "")
            wp = WavPotential(wf)
            wp.plot_radials(0, ax=sp_ax, ax_title="offs = "+str(offs))
            # wp.plot_one_wfm(0)
            # plt.show()

def test_radials(pos, freq, offset):
    wf = tu.transport_waveform([0,0], [1.8,1.8], [1600,1600], 100, "")
    print(wf.samples)
    wp = WavPotential(wf)
    print(wp.find_radials_3d(-1)[0]/1e6)

if __name__ == "__main__":
    # radial_scan([0.2], np.linspace(-1000,1000,25))
    test_radials(0, 1.8, 1600)
    plt.show()
