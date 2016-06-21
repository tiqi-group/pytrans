#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *

# Loading conveyor stuff
from loading_conveyor import *

# Splitting (need to refactor soon - there's a lot of unneeded stuff in splitting.py!)
from splitting import *

def split_waveform():
    pass

def load_and_split(add_reordering=True, analyse_wfms=False):
    wf_path = os.path.join(os.pardir, "waveform_files", "load_split_2016_06_21_v01.dwc.json")
    # If file exists already, just load it to save time
    try:
        # raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load_and_split = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ",wf_path)
        wf_load_path = os.path.join(os.pardir, "waveform_files", "loading_2016_06_21_v01.dwc.json")
        wfs_load = WaveformSet(waveform_file=wf_load_path)

        n_transport = 1000
        # Transport from loading to splitting zone        
        # wf_load_split = transport_waveform(
        #     [0, -422.5], [1.3, 1.356], [960, -2073],
        #     n_transport, "Exp -> left split")
        
        wf_split = split_waveform()
