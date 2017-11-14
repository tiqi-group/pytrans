#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
import copy as cp

# Loading conveyor stuff
import transport_utils as tu
import loading_utils as lu
import splitting as sp

def split_wfms(f_well, conveyor_offs, field_offset, n_transport):
    """ Written at top-level for pickling, so that system can parallelise this """
    return sp.split_waveforms_reparam(0, f_well, conveyor_offs,
                                      [-844, 0], [f_well,f_well], [conveyor_offs, conveyor_offs],
                                      -422.5, f_well,
                                      field_offset=field_offset,
                                      # dc_offset=1000*meV,
                                      n_transport=n_transport,
                                      electrode_subset=[3,4,5,6,7,18,19,20,21,22], # left splitting group
                                      savgol_smooth=False)

def load_and_split_2Be1Ca(add_reordering=True, analyse_wfms=False, save_video=False):
    """ Generate loading/splitting waveforms, with swept offset """
    wf_name = "load_split_2Be1Ca_2017_11_14_v01"
    wf_path = os.path.join(os.pardir, "waveform_files", wf_name + ".dwc.json")

    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load_and_split = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ", wf_path)

        default_freq = 1.1
        default_offs = 1000
        
        # use existing loading conveyor file to save time - need to regenerate if not available
        wfs_load = lu.get_loading_wfms(os.path.join(os.pardir, "waveform_files", "loading_2Be1Ca_2017_11_14_v01.dwc.json"),
                                       default_freq=default_freq,
                                       default_offs=default_offs,
                                       shallow_freq=0.5, shallow_offs=-300, # experimentally optimal for current solver vals
                                       add_reordering=True, ion_chain='2Be1Ca',
                                       force_regen_wfm=True) # Set to True for debug only, otherwise takes a long time!
        
        # truncate waveforms after the first shallow one
        reordering = True
        num_reorder_wfms = 4 if reordering else 0
        if reordering:
            wfs_load_and_split = wfs_load
        else:
            wfs_load_and_split = WaveformSet(
                wfs_load.waveforms[:wfs_load.find_waveform("reorder", get_index=True)-1])
        
        conveyor_offs = default_offs

        n_transport = 500
        f_well = default_freq

        wf_far_to_exp = tu.transport_waveform_multiple(
            [[-844,0],[0,600]],
            [[f_well,f_well],[f_well,f_well]],
            [[conveyor_offs,conveyor_offs],[conveyor_offs,conveyor_offs]],
            2.5*n_transport,
            "-far to centre, centre to +far",
            linspace_fn=zpspace)

        # field_offsets = np.linspace(-300,300,11 - num_reorder_wfms) # Wide scan, to establish what range is reasonable
        # field_offsets = np.linspace(-75, -55, 11-num_reorder_wfms)
        field_offsets = np.linspace(-20, 20, 5)
        # field_offsets = [-63]
        wfs_split = []

        # Solve the wells in parallel (saves a lot of time)

        parallel = False
        if parallel:
            from multiprocessing import Pool
            
            f_wells = np.full_like(field_offsets, f_well)
            transport_offsets = np.full_like(field_offsets, default_offs)
            n_transports = np.full_like(field_offsets, n_transport)
            with Pool(6) as p: # 6 cores:
                res_list = p.starmap(split_wfms, zip(
                    f_wells, transport_offsets, field_offsets, n_transports))
                for centre_to_split, wf_split in res_list:
                    wfs_split.append(wf_split)
        else:
            # serial solver, for debugging
            for field_offset in field_offsets:
                # centre_to_split wastefully defined repeatedly at the moment
                centre_to_split, wf_split = split_wfms(f_well, default_offs, field_offset, n_transport)
                wfs_split.append(wf_split)

        wfs_load_and_split.waveforms.append(centre_to_split)        
        wfs_load_and_split.waveforms += wfs_split       
        wf_recombine_fast = tu.transport_waveform_multiple(
            [[0,0], [600,0]],
            [[f_well,f_well],[f_well,f_well]],
            [[conveyor_offs,conveyor_offs],[conveyor_offs,conveyor_offs]],
            n_transport,
            "recombine, centre, +far -> centre, centre")

        wfs_load_and_split.waveforms.append(wf_far_to_exp)
        wfs_load_and_split.waveforms.append(wf_recombine_fast)
                    
        wfs_load_and_split.write(wf_path, fix_voltage_limits=True)

        st()
        
        if save_video:
            merged_videos = [centre_to_split, wf_split, wf_far_to_exp] # all 3 together
            # merged_videos = [wf_split] # just splitting
            animate_wavpots([WavPotential(k) for k in merged_videos], parallel=False, decimation=1, save_video_path=wf_name+'.mp4')

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
        wfs_load_and_split.write(wf_path, fix_voltage_limits=True)

if __name__ == "__main__":
    load_and_split_2Be1Ca(save_video=True)
