#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *

local_weights = {'r0':1e-5,
                 'r0_u_weights':np.ones(30), # all electrodes uniform
                 'r0_u_ss':np.ones(30)*8,
                 'r1':1e-6,'r2':1e-7}

local_potential_params={'energy_threshold':10*meV}

def plot_selection(pot):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    legends = []
    #pot.plot_range_of_wfms(np.linspace(400,800,100, dtype='int'))
    pot.plot_range_of_wfms(100, ax)
    # for k in range(0,201,10):        
    #     pot.plot_one_wfm(k, ax)
    #     legends.append(str(k))

    plt.legend(legends)
    plt.show()

def static_waveform(pos, freq, offs, wfm_desc, solv_wghts=local_weights):
    wdw = WavDesiredWells([pos*um],[freq*MHz],[offs*meV],

                          solver_weights=solv_wghts,
                          desired_potential_params=local_potential_params,

                          desc=wfm_desc+", {:.3f} MHz, {:.1f} meV".format(freq, offs)
    )
    wf = Waveform(wdw)
    return wf
    
def transport_waveform(pos, freq, offs, timesteps, wfm_desc, linspace_fn=np.linspace):
    # pos, freq, offs: 2-element iterables specifying the start and end, in um, MHz and meV
    wdw = WavDesiredWells(
        [linspace_fn(pos[0], pos[1], timesteps)*um],
        [linspace_fn(freq[0], freq[1], timesteps)*MHz],
        [linspace_fn(offs[0], offs[1], timesteps)*meV],

        solver_weights=local_weights,
        desired_potential_params=local_potential_params,

        desc=wfm_desc+", {:.3f}->{:.3f} MHz, {:.1f}->{:.1f} meV".format(freq[0], freq[1], offs[0], offs[1])
    )
    return Waveform(wdw)

def transport_waveform_multiple(poss, freqs, offsets, timesteps, wfm_desc, linspace_fn = np.linspace):
    # pos, freq, offs: M x 2-element lists specifying M separate
    # wells, like in transport_waveform()
    wdw = WavDesiredWells(
        tuple(linspace_fn(p[0], p[1], timesteps)*um for p in poss),
        tuple(linspace_fn(f[0], f[1], timesteps)*MHz for f in freqs),
        tuple(linspace_fn(o[0], o[1], timesteps)*meV for o in offsets),

        solver_weights=local_weights,
        desired_potential_params=local_potential_params,

        desc=wfm_desc+" {:d} wells".format(len(poss)))
    return Waveform(wdw)

def conveyor_waveform(pos, freq, offs, timesteps, wfm_desc, linspace_fn=np.linspace):
    pts_for_new_wfm = timesteps//5
    conveyor_timesteps = timesteps - pts_for_new_wfm
    offs_ramp_timesteps = (2*conveyor_timesteps)//3
    offs_const_timesteps = conveyor_timesteps - offs_ramp_timesteps
    # Generate the merging section
    wdw = WavDesiredWells(
        [linspace_fn(pos[0], pos[1], conveyor_timesteps)*um, np.ones(conveyor_timesteps)*pos[1]*um],
        [linspace_fn(freq[0], freq[1], conveyor_timesteps)*MHz, np.ones(conveyor_timesteps)*freq[1]*MHz],
        [np.hstack([
            linspace_fn(offs[0], offs[1], offs_ramp_timesteps),
            np.ones(offs_const_timesteps)*offs[1]])*meV, np.ones(conveyor_timesteps)*offs[1]*meV],

        solver_weights=local_weights,
        desired_potential_params=local_potential_params,

        desc=wfm_desc+" conveyor, {:.3f}->{:.3f} MHz, {:.1f}->{:.1f} meV".format(freq[0], freq[1], offs[0], offs[1])
    )
    # Generate the loading well recreation section
    wf = Waveform(wdw)

    if False:
        # Generate a manual waveform whose well goes down from 4200 meV to
        # the initial loading point, manually hacked-in (see potential
        # plots to diagnose level; highly sensitive to waveform solver
        final_loading_pot = 4200 #4230
        wf_rec_dw = WavDesiredWells(
            [np.ones(pts_for_new_wfm)*pos[0]*um, np.ones(pts_for_new_wfm)*pos[1]*um],
            [linspace_fn(0.1, freq[0], pts_for_new_wfm)*MHz, np.ones(pts_for_new_wfm)*freq[1]*MHz],
            [linspace_fn(final_loading_pot, offs[0], pts_for_new_wfm)*meV, np.ones(pts_for_new_wfm)*offs[1]*um],

            solver_weights=local_weights,
            desired_potential_params=local_potential_params,
            desc="")
        wf_rec = Waveform(wf_rec_dw)
        wf.samples = np.hstack([wf.samples, wf_rec.samples])            
    else:
        wf.samples = np.hstack([wf.samples,
                                vlinspace(wf.samples[:,[-1]], wf.samples[:,[0]], pts_for_new_wfm)])
    
    return wf

def reordering_waveform(pos, freq, offs, timesteps, push_v, twist_v, wfm_desc):
    # push_v: voltage by which to increase one side of electrodes
    # twist_v: voltage which is to be differentially applied to twist crystal
    # timesteps: total number of samples for full waveform
    wdw = WavDesiredWells([pos*um], [freq*MHz], [offs*meV],

                          solver_weights=local_weights,
                          desired_potential_params=local_potential_params,

                          desc=wfm_desc+", {:.3f} MHz, {:.1f} meV, push {:.2f} V, twist {:.2f} V".format(freq, offs, push_v, twist_v)
    )
    wf = Waveform(wdw)
    elec_start = wf.samples # vert. array

    def offset_voltages(elec_wfm, electrodes, offsets):
        # elec_wfm: vertical vector with electrode voltages
        # electrodes: array, list or int with electrodes to shift
        # offsets: array or int to shift electrodes by
        if type(offsets) is list:
            offsets = np.array([offsets]).T # to make it 2D
        elec_wfm2 = elec_wfm.copy()
        elec_wfm2[physical_electrode_transform[electrodes]] = elec_wfm[physical_electrode_transform[electrodes]] + offsets
        return elec_wfm2
    
    # Increase potentials on one side of trap, to 'push' out the crystal
    elec_push = offset_voltages(elec_start, [6,7,8], push_v)

    # Differentially twist electrode voltages after push
    elec_push_twist = offset_voltages(elec_push, [6, 8, 21, 23],
                                 [twist_v, -twist_v, -twist_v, twist_v])

    # Undo the push, keep the twist
    elec_twist = offset_voltages(elec_start, [6, 8, 21, 23],
                                 [twist_v, -twist_v, -twist_v, twist_v])
    segment_ts = timesteps//4
    wf.samples = np.hstack([vlinspace(elec_start, elec_push, segment_ts),                            
                            vlinspace(elec_push, elec_push_twist, segment_ts),
                            vlinspace(elec_push_twist, elec_twist, segment_ts),
                            # to avoid rounding changing the number of total samples
                            vlinspace(elec_twist, elec_start, timesteps-3*segment_ts)]) 
    
    return wf
    
def loading_conveyor(add_reordering=True, analyse_wfms=False):
    wf_path = os.path.join(os.pardir, "waveform_files", "loading_2016_07_05_v01.dwc.json")

    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ",wf_path)
        n_load = 1001
        n_freq_change = 200
        default_freq = 1.3
        default_offs = -320

        # List of experimental-zone setting tuples
        exp_settings = [(0, default_freq, default_offs, "exp BeCa")]
        conveyor_offset = 960
        
        wf_load = transport_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offset], n_load, "Load -> exp")
        wf_load_conveyor = conveyor_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offset], n_load, "Load -> exp")
        wf_exp_static_13 = static_waveform(
            0, default_freq, conveyor_offset, "static")
        wf_exp_shallow_13 = transport_waveform(
            [0, 0], [default_freq, 0.3], [conveyor_offset, 0], n_freq_change, "shallow")
        wf_list = [wf_load, wf_load_conveyor,
                   wf_exp_static_13, wf_exp_shallow_13]

        # Default waveform, for reordering
        wf_exp_dual_species = static_waveform(*exp_settings[0])
        
        # Create more deeply confining wells (maybe does not help?)
        deep_weights=dict(local_weights)
        deep_weights['r0'] = 1e-3        
        for pos, freq, offs, label in exp_settings:
            wf_exp_static = static_waveform(
                pos, freq, offs, label)        
            wf_exp_shallow = transport_waveform(
                [pos, pos], [freq, 0.3], [offs, 0], n_freq_change, "shallow")
            wf_exp_static_deep = static_waveform(
                pos, freq, offs, label + " deep", solv_wghts=deep_weights)

            wf_list += [wf_exp_static, wf_exp_shallow, wf_exp_static_deep]

        if add_reordering:
            wf_list += generate_reorder_wfms(wf_exp_dual_species,
                                             [0.4,0.5,0.6,0.7,0.8,1.0,2.0],
                                             [0.4,0.5,0.6,0.7,0.8,1.0,1.5,2.0],
                                             100)
        
        wfs_load = WaveformSet(wf_list)
        wfs_load.write(wf_path)

    if analyse_wfms:
        pot = WavPotential(wfs_load.get_waveform(2))
        pot.plot_one_wfm(0)
        print(pot.find_wells(0, mode='precise'))
        plt.show()

if __name__ == "__main__":
    loading_conveyor(analyse_wfms=False)
