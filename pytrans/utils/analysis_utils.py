#!/usr/bin/env python3
# Various analysis utilities
from pytrans import *

def find_radials(wf, rf_v, timestep=0, plot_results=False, axes=None):
    """Calculate the axial and radial mode frequencies at the centre of
    the trap for this waveform at the timestep given, for each RF
    voltage in the rf_v array (0-peak, in volts). Assumes a single
    ion.
    """
    wfp = WavPotential(wf, rf_v=None)
    N = len(rf_v)
    freqs = np.empty([3, N])
    offsets = np.empty(N)
    for k, v in enumerate(rf_v):
        wfp.rf_v = v
        freqs[:,k], _, _, offsets[k], _ = wfp.find_radials_3d([timestep])

    if plot_results:
        if axes is None:
            f = plt.figure(figsize=(8,11))
            axes = [f.add_subplot(311), f.add_subplot(312), f.add_subplot(313)]

        for ax, freq in zip(axes, freqs):
            ax.plot(rf_v, freq/MHz)
            ax.grid(True)
            ax.set_xlabel('RF voltage')
            ax.set_ylabel('Frequency (MHz)')
        
    return freqs, offsets
