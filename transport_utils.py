#!/usr/bin/env python3
from pytrans import *
import splitting as sp

# default_weights = {'r0':1e-4,
#                    'r0_u_weights':np.ones(30), # all electrodes uniform
#                    'r0_u_ss':np.ones(30)*8,
#                    'r1':3e-5,'r2':3e-6}

# Testing: derivative/2nd deriv/default voltage constraints set to 0
# default_weights = {'r0':1e-15,
#                    'r0_u_weights':np.ones(30), # all electrodes uniform
#                    'r0_u_ss':np.ones(30)*8,
#                    'r1':5e-7,'r2':0}

default_weights = {'r0':5e-10,
                   'r0_u_weights':np.ones(30), # all electrodes uniform
                   'r0_u_ss':np.ones(30)*default_elec_voltage,
                   'r1':5e-4,'r2':2e-6}

default_potential_params={'energy_threshold':2*meV}

def static_waveform(pos, freq, offs, wfm_desc, solv_wghts=default_weights):
    wdw = WavDesiredWells([pos*um],[freq*MHz],[offs*meV],

                          solver_weights=solv_wghts,
                          # solver_weights=default_weights,
                          desired_potential_params=default_potential_params,

                          desc=wfm_desc+", {:.3f} MHz, {:.1f} meV".format(freq, offs)
    )
    wf = Waveform(wdw)
    return wf
    
def transport_waveform(pos, freq, offs, timesteps, wfm_desc,
                       linspace_fn=np.linspace, Ts=200*ns,
                       interp_start=0, interp_end=0):
    # pos, freq, offs: 2-element iterables specifying the start and end, in um, MHz and meV
    if interp_start:
        timesteps -= interp_start
    if interp_end:
        timesteps -= interp_end
    
    wdw = WavDesiredWells(
        [linspace_fn(pos[0], pos[1], timesteps)*um],
        [linspace_fn(freq[0], freq[1], timesteps)*MHz],
        [linspace_fn(offs[0], offs[1], timesteps)*meV],

        solver_weights=default_weights,
        desired_potential_params=default_potential_params,
        Ts=Ts,
        desc=wfm_desc+", {:.3f}->{:.3f} MHz, {:.1f}->{:.1f} meV".format(freq[0], freq[1], offs[0], offs[1])
    )
    wf_wdw = Waveform(wdw)

    if interp_start:
        wdw_start = WavDesiredWells([pos[0]*um], [freq[0]*MHz], [offs[0]*meV],
                                    solver_weights=default_weights,
                                    desired_potential_params=default_potential_params,
                                    Ts=Ts)
        wfs_s = Waveform(wdw_start).samples
        wf_wdw.samples = np.hstack([vlinspace(wfs_s, wf_wdw.samples[:,[0]], interp_start), wf_wdw.samples[:,1:]])
    if interp_end:
        wdw_end = WavDesiredWells([pos[1]*um], [freq[1]*MHz], [offs[1]*meV],
                                    solver_weights=default_weights,
                                    desired_potential_params=default_potential_params,
                                    Ts=Ts)
        wfs_e = Waveform(wdw_end).samples
        wf_wdw.samples = np.hstack([wf_wdw.samples[:,:-1], vlinspace(wf_wdw.samples[:,[-1]], wfs_e, interp_end)])
    
    return wf_wdw

def transport_waveform_multiple(poss, freqs, offsets, timesteps, wfm_desc,
                                linspace_fn=np.linspace, Ts=200*ns,
                                interp_start=0, interp_end=0):
    # pos, freq, offs: M x 2-element lists specifying M separate
    # wells, like in transport_waveform()
    wdw = WavDesiredWells(
        tuple(linspace_fn(p[0], p[1], timesteps)*um for p in poss),
        tuple(linspace_fn(f[0], f[1], timesteps)*MHz for f in freqs),
        tuple(linspace_fn(o[0], o[1], timesteps)*meV for o in offsets),

        solver_weights=default_weights,
        desired_potential_params=default_potential_params,
        Ts=Ts,
        desc=wfm_desc+" {:d} wells".format(len(poss)))    
    wf_wdw = Waveform(wdw)

    if interp_start:
        wdw_start = WavDesiredWells(tuple(p[0]*um for p in poss),
                                    tuple(f[0]*MHz for f in freqs),
                                    tuple(o[0]*meV for o in offsets),
                                    solver_weights=default_weights,
                                    desired_potential_params=default_potential_params,
                                    Ts=Ts)
        wfs_s = Waveform(wdw_start).samples
        wf_wdw.samples = np.hstack([vlinspace(wfs_s, wf_wdw.samples[:,[0]], interp_start), wf_wdw.samples[:,1:]])
    if interp_end:
        wdw_end = WavDesiredWells(tuple(p[1]*um for p in poss),
                                    tuple(f[1]*MHz for f in freqs),
                                    tuple(o[1]*meV for o in offsets),
                                    solver_weights=default_weights,
                                    desired_potential_params=default_potential_params,
                                    Ts=Ts)
        wfs_e = Waveform(wdw_end).samples
        wf_wdw.samples = np.hstack([wf_wdw.samples[:,:-1], vlinspace(wf_wdw.samples[:,[-1]], wfs_e, interp_end)])
    
    return wf_wdw

def conveyor_rec_waveform(loc, freq, offs, timesteps, wfm_desc, linspace_fn=np.linspace):
    """Similar purpose to conveyor_waveform, but aim is to recombine
    wells in the splitting zone using optimal solver"""
    load_to_rec_ts = timesteps//4 #loading -> recombination start transport
    rec_ts = timesteps//4 # recombine -> end up in splitting zone [WARNING: NOT USED FOR NOW]
    split_to_exp_ts = timesteps//4 # splitting -> experimental transport

    load_loc = loc[0]
    load_freq = freq[0]
    load_offs = offs[0]

    exp_loc = loc[1]
    exp_freq = freq[1]
    exp_offs = offs[1]
    
    rec_loc = -844
    # interpolate recombine params
    rec_ratio = (exp_loc - rec_loc)/(exp_loc - load_loc)
    rec_freq = exp_freq + (load_freq - exp_freq) * rec_ratio
    rec_offs = exp_offs + (load_offs - exp_offs) * rec_ratio
    
    split_loc = -422.5
    # interpolate split params
    split_ratio = (exp_loc - split_loc)/(exp_loc - load_loc)
    split_freq = exp_freq + (load_freq - exp_freq) * split_ratio
    split_offs = exp_offs + (load_offs - exp_offs) * split_ratio
    
    # (Load, exp) -> (rec, exp)
    wf_load_to_rec = transport_waveform_multiple(
        [[load_loc, rec_loc], [exp_loc, exp_loc]],
        [[load_freq, rec_freq], [exp_freq, rec_freq]],
        [[load_offs, rec_offs], [exp_offs, rec_offs]],
        load_to_rec_ts,
        "",
        linspace_fn=zpspace)
        #interp_start=50, interp_end=50)

    wf_exp_to_split, wf_split = sp.split_waveforms_reparam(
        exp_loc, exp_freq, exp_offs,
        [rec_loc, exp_loc],
        [rec_freq, rec_freq],
        [rec_offs, rec_offs],
        split_loc, split_freq, None,
        split_to_exp_ts,
        electrode_subset=[3,4,5,6,7,18,19,20,21,22],
        savgol_smooth=False)

    total_samples = np.hstack([
        wf_load_to_rec.samples, np.fliplr(wf_split.samples), np.fliplr(wf_exp_to_split.samples)])

    # Recreate loading-zone well
    total_samples = np.hstack([total_samples, vlinspace(total_samples[:,[-1]], total_samples[:,[0]], timesteps//5, lin_fn=zpspace) ])
    
    wf_load_conveyor = Waveform(
        wfm_desc+" conveyor, {:.3f}->{:.3f} MHz, {:.1f}->{:.1f} meV".format(
            load_freq, exp_freq, load_offs, exp_offs),
        0, "", total_samples)

    return wf_load_conveyor

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

        solver_weights=default_weights,
        desired_potential_params=default_potential_params,

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

            solver_weights=default_weights,
            desired_potential_params=default_potential_params,
            desc="")
        wf_rec = Waveform(wf_rec_dw)
        wf.samples = np.hstack([wf.samples, wf_rec.samples])            
    else:
        wf.samples = np.hstack([wf.samples,
                                vlinspace(wf.samples[:,[-1]], wf.samples[:,[0]], pts_for_new_wfm)])
    
    return wf

def shallow_waveform(freqs, dc_offsets, n_steps=101, lr_offset=0, tb_offset=0):
    shallow_wfm = transport_waveform([0,0], freqs, dc_offsets,
                                     n_steps,
                                     "shallow with T-B offs {:.3f}, L-R offs {:.3f}".format(
                                         tb_offset, lr_offset), linspace_fn=zpspace)
    lr_offset_vec = np.vstack([np.full([7,1], lr_offset/2), 0, np.full([7,1], -lr_offset/2)]) # 15//2 = 7 (half the trap)
    top_offset_vec = lr_offset_vec + tb_offset/2
    bot_offset_vec = lr_offset_vec - tb_offset/2

    shallow_wfm.samples[physical_electrode_transform,:] += vlinspace(
        np.zeros([30,1]), np.vstack([top_offset_vec, bot_offset_vec]), n_steps)

    return shallow_wfm    

def reordering_waveform(pos, freq, offs, timesteps, push_v, twist_v, wfm_desc):
    # push_v: voltage by which to increase one side of electrodes
    # twist_v: voltage which is to be differentially applied to twist crystal
    # timesteps: total number of samples for full waveform
    wdw = WavDesiredWells([pos*um], [freq*MHz], [offs*meV],

                          solver_weights=default_weights,
                          desired_potential_params=default_potential_params,

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
