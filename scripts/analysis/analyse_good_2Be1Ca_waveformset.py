#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
import copy as cp

# Loading conveyor stuff
import transport_utils as tu
import loading_utils as lu

wf_path_orig = os.path.join(os.pardir, "waveform_files", "load_split_2Be1Ca_2017_02_21_v04.dwc.json")
wf_path_new = os.path.join(os.pardir, "waveform_files", "load_split_2Be1Ca_2017_02_21_v04_mod_v01.dwc.json")

def get_pos(wf):
    wfp = WavPotential(wf)
    wfp_fit = wfp.find_wells(-1, mode='precise', roi_width=30*um)
    loc_um = wfp_fit['locs'][0]*1e6
    return loc_um

def fix_electrodes(wf, initial_step=0.001, electrodes_to_adjust=[9,24]):
    """ Fix effect of 'bad' electrode and its mirror-image counterpart """

    old_loc = get_pos(wf)
    wf.samples[physical_electrode_transform[electrodes_to_adjust]] += initial_step    
    new_loc = get_pos(wf)
    desired_loc = 0*1e6 # where the location should be adjusted to
    loc_per_v = (new_loc - old_loc)/initial_step
    old_loc = get_pos(wf)
    for k in range(10):
        # Very simple iterative loop to minimise position offset
        print("Pos before loop {:d}: {:f}".format(k, old_loc))
        wf.samples[physical_electrode_transform[electrodes_to_adjust]] += (desired_loc - old_loc)/loc_per_v
        new_loc = get_pos(wf)
        old_loc = new_loc

    print("Pos after fix: {:f}".format(new_loc))

def fix_static_wf(wf, electrodes_to_shift=[]):
    print("Fixing Waveform: {:s}".format(wf.desc))
    wf.samples[physical_electrode_transform[10]] = 0 # effect of 'dead' electrode 10
    print("Uncorrected position: {:f} um".format(get_pos(wf)))
    wf.samples[physical_electrode_transform[[25]]] = 0 # effect of dead electrode's partner
    # wf.samples[physical_electrode_transform[[4, 19]]] = 0 # effect of 'dead' electrode 10, and its partner
    fix_electrodes(wf)
        
def analyse_well():
    wfs = WaveformSet(waveform_file=wf_path_orig)
    print("Loaded waveform ",wf_path_orig)

    wflist = wfs.find_waveforms('static')
    for wf in wflist:
        fix_static_wf(wf)
    
    wflist = wfs.find_waveforms('shallow')
    for wf in wflist:
        fix_static_wf(wf)
    
    wfp = WavPotential(wf) # waveform potential

    # fit to the well found within the ROI, print its properties
    wp = wfp.find_wells(-1, mode='precise', roi_width=50*um) 
    print("Shallow well: freq {:.3f} MHz, offset {:.3f} mV, loc {:.3f} um".format(
        wp['freqs'][0]/1e6, wp['offsets'][0]*1e3, wp['locs'][0]*1e6))

    print(get_pos(wfs.find_waveforms('shallow')[1]))
    
    wfs.write(wf_path_new)
    print("Wrote waveform ",wf_path_new)
    
    wfp.plot_one_wfm(-1)
    plt.show()

if __name__ == "__main__":
    analyse_well()
