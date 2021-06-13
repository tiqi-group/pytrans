#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
import copy as cp

# Loading conveyor stuff
import transport_utils as tu
import loading_utils as lu

wf_path = os.path.join(os.pardir, "waveform_files", "static_well_2Be1Ca_2017_07_17_v01.dwc.json")

def static_well():
    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ", wf_path)

        wf = tu.static_waveform(0, 1.6, 1000, "Test")
        wf_list = [wf]

        # (OPTIONAL) Generate extra waveforms
        for k in range(2,10):
            wf = tu.static_waveform(0, 1.6 + 0.05*(k-1), 1000, "Test {:d}".format(k))
            wf_list.append(wf)
        
        wfs = WaveformSet(wf_list)
        wfs.write(wf_path)

def plot_well():
    wfs = WaveformSet(waveform_file=wf_path)
    print("Loaded waveform ",wf_path)

    wf = wfs.find_waveform('Test') # the string just needs to be in the waveform name
    wfp = WavPotential(wf) # waveform potential

    # fit to the well found within the ROI, print its properties
    wp = wfp.find_wells(0, mode='precise', roi_width=50*um) 
    print("Generated well: freq {:.3f} MHz, offset {:.3f} mV, loc {:.3f} um".format(
        wp['freqs'][0]/1e6, wp['offsets'][0]*1e3, wp['locs'][0]*1e6))

    # (OPTIONAL) Analyse extra waveforms
    print("Extra waveforms:")
    for wf in wfs.waveforms:
        wp = WavPotential(wf).find_wells(0, mode='precise', roi_width=50*um)
        print (len(wp['freqs']))
        for i in range(len(wp['freqs'])):
            print('Desc: "{:s}", fitted freq {:.3f} MHz, offset {:.3f} mV, loc {:.3f} um'.format(
                wf.desc, wp['freqs'][i]/1e6, wp['offsets'][i]*1e3, wp['locs'][i]*1e6))

if __name__ == "__main__":
    static_well() # generate waveform, save into waveform file
    plot_well() # open waveform file and plot waveform
