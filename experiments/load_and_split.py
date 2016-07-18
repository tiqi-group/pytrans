#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
import copy as cp

# Loading conveyor stuff
import loading_conveyor as lc

# Splitting (need to refactor soon - there's a lot of unneeded stuff in splitting.py!)
import splitting as sp

def merge_waveforms_for_rev(wfs):
    samples_forward = np.hstack(wf.samples for wf in wfs)
    samples_for_rev = np.hstack([samples_forward, np.fliplr(samples_forward)])
    wf_for_rev = Waveform("forward, then reverse", 0, "", samples_for_rev)
    wf_for_rev.set_new_uid()
    return wf_for_rev

def split_waveforms(
        start_loc, start_f, start_offset,
        final_locs, final_fs, final_offsets,
        split_loc, split_f, split_offset=None,
        n_transport=2000,        
        field_offset = 0,
        electrode_subset=None,
        start_split_label='trans from start -> split start',
        split_label='split apart',
        plot_splits=False):
    # Specify the starting well properties (the experimental zone
    # usually) and the splitting well properties, which will be used
    # in the linear ramp between the combined well and the first stage
    # of the quartic polynomial solver.
    # Note: final_locs, final_freqs, final_offsets
    split_centre = split_loc*um # centre of the central splitting electrode moment
    polyfit_range = 200*um

    polys = sp.generate_interp_polys(trap_mom.transport_axis,
                                    trap_mom.potentials[:, electrode_subset],
                                    split_centre, polyfit_range)
    
    # Data format is (alpha, slope, points from prev. state to this one)
    # Requires careful tuning

    # field_offset = -33.5 # 2Ca splitting
    # field_offset = -35
    # field_offset = -10 # 2Be splitting
    
    interp_steps = 75
    split_params = [# (1.5e7, None, 500, np.linspace),
        # (1e6, None, 500, np.linspace),
        #(0, field_offset, 500, lambda a,b,n: erfspace(a,b,n,1.5)),
#        (1e6, field_offset, 200, np.linspace), # TODO: uncomment this
#        (1e6, field_offset, interp_steps//2, np.linspace),        
        (5e5, field_offset, interp_steps, np.linspace),
        (0, field_offset, interp_steps//2, np.linspace),
        # (-3e6, None, 500, np.linspace),
        (-2.5e6, field_offset, interp_steps//2, np.linspace),
        (-5e6, field_offset, interp_steps//2, np.linspace),
        (-7.5e6, field_offset, interp_steps//2, np.linspace),        
        (-1e7, field_offset, interp_steps//2, np.linspace),
        (-1.5e7, field_offset, interp_steps//2, np.linspace)]
        # (-2e7, None, 50, np.linspace),
        # (-3e7, None, interp_steps, np.linspace),
        # (-4e7, None, interp_steps, np.linspace),
        # (-5e7, None, 150, np.linspace),
        # (-6e7, None, 300, np.linspace)]

    if not split_offset:
        # automatically figure out the potential offset by running the
        # solver for the initial splitting conditions and fitting to it
        death_v_set = np.zeros([num_elecs, 1])
        sp_start = split_params[0]
        elec_v_set,_,_ = sp.solve_poly_ab(polys, alpha=sp_start[0],
                                          slope_offset=sp_start[1])
        death_v_set[physical_electrode_transform[electrode_subset]] = elec_v_set
        wavpot_fit = find_wells_from_samples(death_v_set,
                                             roi_centre=split_centre,
                                             roi_width=polyfit_range)
        assert len(wavpot_fit['offsets']) == 1, "Error, found too many wells in ROI at start of splitting."
        split_offset = wavpot_fit['offsets'][0]/meV
        
    # Initial waveform, transports from start to splitting location
    wf_split = lc.transport_waveform(
        [start_loc, split_loc],
        [start_f, split_f],
        [start_offset, split_offset], n_transport, start_split_label)
        
    latest_death_voltages = wf_split.samples[:,[-1]] # square bracket to return column vector
    full_wfm_voltages = latest_death_voltages.copy()

    plot_splitting_parts = False
    # Prepare full voltage array
    for (alpha, slope_offset, npts, linspace_fn) in split_params:
        elec_voltage_set,alpha,beta = sp.solve_poly_ab(polys, alpha,
                                            slope_offset=slope_offset, dc_offset=None)
        new_death_voltages = latest_death_voltages.copy()
        new_death_voltages[physical_electrode_transform[electrode_subset]] = elec_voltage_set

        # Ramp from old to new voltage set
        ramped_voltages = vlinspace(latest_death_voltages, new_death_voltages,
                                    npts, linspace_fn)[:,1:]
        full_wfm_voltages = np.hstack([full_wfm_voltages, ramped_voltages])
        latest_death_voltages = new_death_voltages
        
        if plot_splitting_parts:
            new_wf = Waveform("", 0, "", ramped_voltages)
            asdf = WavPotential(new_wf)
            asdf.plot_range_of_wfms(20)
            # plt.set_ylim([-1.470,-1.435])
            # plt.set_xlim([-500,-350])            
            plt.show()

    npts_final = npts

    final_splitting_params = find_wells_from_samples(
        latest_death_voltages, split_centre, polyfit_range)
    split_locs = np.array(final_splitting_params['locs'])/um
    split_freqs = np.array(final_splitting_params['freqs'])/MHz
    split_offsets = np.array(final_splitting_params['offsets'])/meV
    assert split_locs.size == 2, "Wrong number of wells detected after splitting"

                                                                     
    # Final waveform, extends separation by 150um either way and goes to default well settings
    # (starting values must be set to the results of the splitting!)
    wf_finish_split = lc.transport_waveform_multiple(
        [[split_locs[0], final_locs[0]],[split_locs[1], final_locs[1]]],
        [[split_freqs[0],final_fs[0]],[split_freqs[1],final_fs[1]]],
        [[split_offsets[0], final_offsets[0]],[split_offsets[1], final_offsets[1]]],
        n_transport,
        "")
    
    # Remove final segment of full voltage array, replace with manual
    # ramp to start of regular solver
    final_ramp_start = full_wfm_voltages[:,[-npts_final]]
    final_ramp_end = wf_finish_split.samples[:,[0]]
    full_wfm_voltages = full_wfm_voltages[:,:-npts_final+1] # final_ramp_start voltage set

    final_ramped_voltages = vlinspace(final_ramp_start, final_ramp_end, npts_final*3//4, linspace_fn)[:,1:]
    full_wfm_voltages = np.hstack([full_wfm_voltages, final_ramped_voltages])

    # Append final splitting wfm
    full_wfm_voltages = np.hstack([full_wfm_voltages, wf_finish_split.samples[:,1:]])

    savgol_smooth = True
    # Smooth the waveform voltages
    if savgol_smooth:
        full_wfm_voltages_filt = ssig.savgol_filter(full_wfm_voltages, 151, 2, axis=-1)
    else:
        full_wfm_voltages_filt = full_wfm_voltages

    spline_smooth = False
    if spline_smooth:
        t_axis = np.arange(full_wfm_voltages.shape[1])        
        fwv_funcs = (sintp.UnivariateSpline(t_axis, fwv, s=10, k=5) for fwv in full_wfm_voltages)
        full_wfm_voltages_filt = np.vstack((fwv_func(t_axis) for fwv_func in fwv_funcs))
            
    # splitting_wf = Waveform(split_label, 0, "", full_wfm_voltages)
    splitting_wf = Waveform(split_label+", offset = " + str(field_offset*um) + " V/m",
                            0, "", full_wfm_voltages_filt)
    splitting_wf.set_new_uid()
    
    if False:
        asdf = WavPotential(splitting_wf)
        print(asdf.find_wells(-1))
        asdf.plot_one_wfm(-1)
        plt.show()
    animate_waveform = False
    if animate_waveform:
        # Set up formatting for the movie files
        Writer = anim.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_ylim([-4,4])
        line, = ax.plot(asdf.trap_axis/um, asdf.potentials[:,0])
        def update(data):
            line.set_ydata(data)
            return line
        
        def data_gen():
            for pot in asdf.potentials.T[::10]:
                yield pot

        # im_ani = anim.ArtistAnimation(plt.figure(), ims, interval=100, repeat_delay=5000, blit=True)
        
        im_ani = anim.FuncAnimation(fig, update, data_gen, interval=30)

        plt.show()
        im_ani.save('im.mp4', writer=writer)

    return wf_split, splitting_wf

def load_and_split(add_reordering=True, analyse_wfms=False):
    wf_path = os.path.join(os.pardir, "waveform_files", "load_split_2016_07_15_v01.dwc.json")
    # If file exists already, just load it to save time
    try:
        raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load_and_split = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ",wf_path)
        # use existing loading conveyor file to save time - need to regenerate if not available
        wf_load_path = os.path.join(os.pardir, "waveform_files", "loading_2016_07_15_v01.dwc.json")
        wfs_load = WaveformSet(waveform_file=wf_load_path)
        wfs_load_and_split = wfs_load

        conveyor_offset = 960

        n_transport = 308

        wf_far_to_exp = lc.transport_waveform_multiple(
            [[-844,0],[0,600]],
            [[1.3,1.3],[1.3,1.3]],
            [[conveyor_offset,conveyor_offset],[conveyor_offset,conveyor_offset]],
            2.5*n_transport,
            "-far to centre, centre to +far")
        
        field_offsets = np.linspace(-29.5,-24.5,11)
        wfs_split = []
        for field_offset in field_offsets:
            load_to_split, wf_split = split_waveforms(0, 1.3, conveyor_offset,
                                                      [-844, 0], [1.3,1.3], [conveyor_offset, conveyor_offset],
                                                      -422.5, 1.3,
                                                      field_offset=field_offset,
                                                      n_transport=n_transport,
                                                      electrode_subset=[3,4,5,6,7,18,19,20,21,22]) # left splitting group

            
            # Interpolate between end of splitting and start of parallel transport
            split_trans_interp = vlinspace(wf_split.samples[:,[-1]], wf_far_to_exp.samples[:,[0]], 50)
            wf_split.samples = np.hstack((wf_split.samples, split_trans_interp))
            wfs_split.append(wf_split)

        wfs_load_and_split.waveforms.append(load_to_split)        
        wfs_load_and_split.waveforms += wfs_split       
        wf_recombine_fast = lc.transport_waveform_multiple(
            [[0,0], [600,0]],
            [[1.3,1.3],[1.3,1.3]],
            [[conveyor_offset,conveyor_offset],[conveyor_offset,conveyor_offset]],
            n_transport,
            "recombine, centre, +far -> centre, centre")

        wfs_load_and_split.waveforms.append(wf_far_to_exp)
        wfs_load_and_split.waveforms.append(wf_recombine_fast)

        animate_split = False
        if animate_split:
            animate_wavpots([WavPotential(k) for k in (load_to_split, wf_split, wf_far_to_exp)], parallel=False, decimation=1)# , save_video_path='load_and_split.mp4')
        
            
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
    load_and_split()
