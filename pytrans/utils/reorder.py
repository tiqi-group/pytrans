#!/usr/bin/env python3

import sys
# sys.path.append("../")
from pytrans.pytrans import *
from . import transport_utils as tu
import copy as cp

# This script can be used to append multiple reorder operations to an existing waveform file

def offset_voltages(elec_wfm, electrodes, offsets):
    # elec_wfm: vertical vector with electrode voltages
    # electrodes: array, list or int with electrodes to shift
    # offsets: array or int to shift electrodes by
    if type(offsets) is list:
        offsets = np.array([offsets]).T # to make it 2D
    elec_wfm2 = elec_wfm.copy()
    elec_wfm2[physical_electrode_transform[electrodes]] = elec_wfm[physical_electrode_transform[electrodes]] + offsets
    return elec_wfm2

def generate_reorder_wfms(wf, push_v_vec=[0.3], twist_v_vec=[0.5], timesteps=100):
    elec_start = wf.samples # vert. array
    assert wf.samples.shape == (32,1), "Please supply a static waveform for the reordering"
    new_wfs = []
    for pv in push_v_vec:
        for tv in twist_v_vec:
            elec_push = offset_voltages(elec_start, [6,7,8], pv)
            elec_push_twist = offset_voltages(elec_push, [6,8,21,23], [tv,-tv,-tv,tv])
            elec_twist = offset_voltages(elec_start, [6,8,21,23], [tv,-tv,-tv,tv])

            new_wf = cp.deepcopy(wf)
            segment_ts = timesteps//4
            new_wf.samples = np.hstack([vlinspace(elec_start, elec_push, segment_ts),
                                        vlinspace(elec_push, elec_push_twist, segment_ts),
                                        vlinspace(elec_push_twist, elec_twist, segment_ts),
                                        vlinspace(elec_twist, elec_start, segment_ts)])
            if new_wf.voltage_limits_exceeded():
                print("push {:.1f} V, twist {:.1f} V exceeds DAC voltage limits".format(pv, tv))
            else:
                new_wf.set_new_uid()
                new_wf.desc += " reorder: push {:.1f} V twist {:.1f} V".format(pv, tv)
                # st()
                new_wfs.append(new_wf)

    return new_wfs

def generate_reorder_2Be2Ca_wfms(start_freq, start_offs, target_freq_vec=[1.0], target_offs_vec=[1.0], timesteps_vec=[500]):
    # elec_start = wf.samples # vert. array
    # assert wf.samples.shape == (32,1), "Please supply a static waveform for the reordering"
    new_wfms = []
    for freq, offs, timesteps in zip(target_freq_vec, target_offs_vec, timesteps_vec):
        new_wf = tu.transport_waveform([0,0], [start_freq, freq], [start_offs, offs],
                               timesteps, "2Be2Ca reorder")
        new_wf.samples = np.hstack([new_wf.samples, np.fliplr(new_wf.samples)])
        new_wfms.append(new_wf)
    return new_wfms

if __name__ == "__main__":
    wf_path_conveyor = os.path.join(os.pardir, "waveform_files", "loading_conveyor_2016_06_08_v01.dwc.json")
    wfs = WaveformSet(waveform_file=wf_path_conveyor)
    wf = wfs.get_waveform(5) # should be a static waveform

    # Generate a bunch of new reordering waveforms
    wfs.waveforms += generate_reorder_wfms(wf, [0.1,0.4,0.7,1.0,1.5,2.3,3,4], [0.1,0.4,0.7,1.0,1.5,2.3,3,4], 100)

    # Combine several waveforms into one
    # wfs.waveforms += combine_recovery_wfms(wfs.get_waveform

    wf_path_conveyor_new = os.path.join(os.pardir, "waveform_files", "loading_conveyor_reorder_2016_06_08_v01.dwc.json")
    wfs.write(wf_path_conveyor_new)
