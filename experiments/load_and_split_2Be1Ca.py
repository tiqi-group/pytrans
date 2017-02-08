#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
import copy as cp

# Loading conveyor stuff
import transport_utils as tu
import loading_utils as lu
import splitting as sp

def load_and_split_2Be1Ca(add_reordering=True, analyse_wfms=False):
    """ Generate loading/splitting waveforms, with swept offset """
    wf_path = os.path.join(os.pardir, "waveform_files", "load_split_2Be1Ca_2017_02_08_v04.dwc.json")

    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load_and_split = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ", wf_path)
        # use existing loading conveyor file to save time - need to regenerate if not available
        wfs_load = lu.get_loading_wfms("loading_2Be1Ca_2017_01_25_v01.dwc.json",
                                       add_reordering=True, ion_chain='2Be1Ca')
        
        # truncate waveforms after the first shallow one
        reordering = True
        num_reorder_wfms = 4 if reordering else 0
        if reordering:
            wfs_load_and_split = wfs_load
        else:
            wfs_load_and_split = WaveformSet(
                wfs_load.waveforms[:wfs_load.find_waveform("shallow", get_index=True)+1])

        default_freq = 1.1
        default_offs = 1000
        
        conveyor_offset = default_offs

        n_transport = 308
        interp_n = n_transport//10
        f_well = default_freq

        wf_far_to_exp = tu.transport_waveform_multiple(
            [[-844,0],[0,600]],
            [[f_well,f_well],[f_well,f_well]],
            [[conveyor_offset,conveyor_offset],[conveyor_offset,conveyor_offset]],
            2.5*n_transport,
            "-far to centre, centre to +far",
            interp_start=40, interp_end=40)

        # field_offsets = np.linspace(-300,300,11 - num_reorder_wfms) # Wide scan, to establish what range is reasonable
        field_offsets = np.linspace(-70, 30, 11-num_reorder_wfms)
        wfs_split = []
        for field_offset in field_offsets:
            centre_to_split, wf_split = sp.split_waveforms(0, f_well, conveyor_offset,
                                                           [-844, 0], [f_well,f_well], [conveyor_offset, conveyor_offset],
                                                           -422.5, f_well,
                                                           field_offset=field_offset,
                                                           n_transport=n_transport,
                                                           electrode_subset=[3,4,5,6,7,18,19,20,21,22]) # left splitting group

            
            # Interpolate between end of splitting and start of parallel transport
            split_trans_interp = vlinspace(wf_split.samples[:,[-1]], wf_far_to_exp.samples[:,[0]], interp_n)
            wf_split.samples = np.hstack((wf_split.samples, split_trans_interp))
            wfs_split.append(wf_split)

        wfs_load_and_split.waveforms.append(centre_to_split)        
        wfs_load_and_split.waveforms += wfs_split       
        wf_recombine_fast = tu.transport_waveform_multiple(
            [[0,0], [600,0]],
            [[f_well,f_well],[f_well,f_well]],
            [[conveyor_offset,conveyor_offset],[conveyor_offset,conveyor_offset]],
            n_transport,
            "recombine, centre, +far -> centre, centre")

        wfs_load_and_split.waveforms.append(wf_far_to_exp)
        wfs_load_and_split.waveforms.append(wf_recombine_fast)

        animate_split = False
        if animate_split:
            animate_wavpots([WavPotential(k) for k in (centre_to_split, wf_split, wf_far_to_exp)], parallel=False, decimation=1, save_video_path='load_and_split.mp4')
        
            
        wfs_load_and_split.write(wf_path)

    # Create a single testing waveform, made up of the individual transports
    add_testing_waveform = False
    if add_testing_waveform:
        test_waveform_present = wfs_load_and_split.get_waveform(-1).desc == "trans + split, then reverse"
        if test_waveform_present:
            exp_to_split_wfm = wfs_load_and_split.get_waveform(-4)
            split_wfm = wfs_load_and_split.get_waveform(-3)
            far_to_cent_wfm = wfs_load_and_split.get_waveform(-2)
        else:
            exp_to_split_wfm = wfs_load_and_split.get_waveform(-3)
            split_wfm = wfs_load_and_split.get_waveform(-2)
            far_to_cent_wfm = wfs_load_and_split.get_waveform(-1)            

        #trans_split_forward = np.hstack([exp_to_split_wfm.samples, split_wfm.samples[:,:-500]])
        trans_split_forward = np.hstack([exp_to_split_wfm.samples, split_wfm.samples, far_to_cent_wfm.samples])
        trans_split_for_rev = np.hstack([trans_split_forward, np.fliplr(trans_split_forward)])
        wf_trans_split_for_rev = Waveform("trans + split, then reverse", 0, "", trans_split_for_rev)
        wf_trans_split_for_rev.set_new_uid()

        if test_waveform_present:
            wfs_load_and_split.waveforms[-1] = wf_trans_split_for_rev
        else:
            wfs_load_and_split.waveforms.append(wf_trans_split_for_rev)

        wf_forward = Waveform("", 0, "", trans_split_forward)
        pot_forward = WavPotential(wf_forward)
        pot_for_rev = WavPotential(wf_trans_split_for_rev)

        print(pot_forward.find_wells(-1))
        pot_for_rev.animate_wfm(decimation=1)
        wfs_load_and_split.write(wf_path)

if __name__ == "__main__":
    load_and_split_2Be1Ca()
