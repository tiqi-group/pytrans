#!/usr/bin/env python3
# Loading library, used to produce loading conveyor waveforms etc

import sys
sys.path.append("./")
from pytrans import *
import transport_utils as tu
import reorder as ror

def get_loading_wfms(wfm_path, force_regen_wfm=False,
                     default_freq=1.1, default_offs=1000, # used for static and return-from-shallow
                     # shallow_freq=0.3, shallow_offs=-550, # used for shallow well
                     shallow_freq=0.3, shallow_offs=-550, # used for shallow well
                     add_reordering=False,
                     add_deep_wells=False,
                     analyse_wfms=False,
                     ion_chain='2Be1Ca'):
    # If file exists already, just load it to save time
    try:
        # raise FileNotFoundError # uncomment to always regenerate file for debugging
        if force_regen_wfm:
            raise FileNotFoundError
        wfs_load = WaveformSet(waveform_file=wfm_path)
        print("Opened loading waveform ",wfm_path)
    except FileNotFoundError:
        print("Generating loading waveform ",wfm_path)
        n_load = 1001
        n_freq_change = 101

        # List of experimental-zone setting tuples
        exp_settings = [(0, default_freq, default_offs, "exp " + ion_chain)]
        conveyor_offs = default_offs
        
        wf_load = tu.transport_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offs], n_load, "Load -> exp", linspace_fn=zpspace)
        if False:
            conv_fn = tu.conveyor_rec_waveform # recombine section using polynomial solver
        else:
            conv_fn = tu.conveyor_waveform # recombine section using regular solver
        wf_load_conveyor = conv_fn(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offs], 2*n_load, "Load -> exp", linspace_fn=zpspace)
        wf_exp_static = tu.static_waveform(
            0, default_freq, conveyor_offs, "static")
        wf_list = [wf_load, wf_load_conveyor, wf_exp_static]
        # wf_exp_shallow = tu.transport_waveform(
        #     [0, 0], [default_freq, shallow_freq], [conveyor_offs, shallow_offs], n_freq_change, "shallow", linspace_fn=zpspace)
        # wf_list.append(wf_exp_shallow)

        # Generate shallow wells with a range of compensation forces; left-vs-right and top-vs-bottom.
        # lr_offsets = np.linspace(-0.2, 0.2, 11)
        # tb_offsets = np.linspace(-0.2, 0.2, 11)
        lr_offsets = [-0.08]
        tb_offsets = [-0.08]
        for lr_offset in lr_offsets:
            for tb_offset in tb_offsets:
                shallow_wfm = tu.transport_waveform(
                    [0,0],
                    [default_freq, shallow_freq],
                    [conveyor_offs, shallow_offs],
                    n_freq_change, "shallow with T>B offs {:.3f}, L>R offs {:.3f}".format(tb_offset, lr_offset), linspace_fn=zpspace
                    )
                lr_offset_vec = np.vstack([np.full([7,1], lr_offset/2), 0, np.full([7,1], -lr_offset/2)])
                top_offset_vec = lr_offset_vec + tb_offset/2
                bot_offset_vec = lr_offset_vec - tb_offset/2

                shallow_wfm.samples[physical_electrode_transform,:] += vlinspace(
                    np.zeros([30,1]), np.vstack([top_offset_vec, bot_offset_vec]), n_freq_change)
                wf_list.append(shallow_wfm)

        # for offs in lr_offsets:
        #     for level in ['top', 'bottom']:
        #         shallow_wfm = tu.transport_waveform(
        #             [0,0],
        #             [default_freq, 0.4],
        #             [conveyor_offs, -300],
        #             n_freq_change, "shallow with T>B offs {:.3f}, L>R offs {:.3f}".format(offs_tb, offs_lr), linspace_fn=zpspace)
        #         if side == 'left':
        #             offsets = np.vstack([np.full([7,1], offs), np.zeros([8,1])])
        #         else:
        #             offsets = np.vstack([np.zeros([8,1]), np.full([7,1], offs)])
        #         if level == 'top':
        #             top_offset = offsets
        #             bot_offset = -offsets
        #         else:
        #             top_offset = -offsets
        #             bot_offset = offsets

        #         shallow_wfm.samples[physical_electrode_transform,:] += vlinspace(
        #             np.zeros([30,1]), np.vstack([top_offset, bot_offset]), n_freq_change)
        #         wf_list.append(shallow_wfm)
                    

        analyse_static_radials = False
        if analyse_static_radials:
            n_timesteps = wf_exp_shallow.samples.shape[1]
            vec_mode1 = np.zeros((n_timesteps,2))
            vec_mode2 = np.zeros_like(vec_mode1)
            
            for k in range(n_timesteps):
                omegas, axes, r0, offset, V = analyse_wfm_radials(wf_exp_shallow, -1, False)
                vec_mode1[k, 0:2] = omegas[0]*axes[1,1:3]
                vec_mode2[k, 0:2] = omegas[1]*axes[2,1:3]

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            plt.plot(np.arange(n_timesteps), vec_mode1[:,0], vec_mode1[:,1], 'r')
            plt.plot(np.arange(n_timesteps), vec_mode2[:,0], vec_mode2[:,1], 'g')
            plt.show()
            # analyse_wfm_radials(wf_exp_static, 0)
        
        # Default waveform, for reordering
        wf_exp_dual_species = tu.static_waveform(*exp_settings[0])
        
        # Create more deeply confining wells (maybe does not help?)
        deep_weights=dict(tu.default_weights)
        deep_weights['r0'] = 1e-3

        if False:
            for pos, freq, offs, label in exp_settings:
                wf_exp_static = tu.static_waveform(
                    pos, freq, offs, label)
                wf_exp_shallow = tu.transport_waveform(
                    [pos, pos], [freq, shallow_freq], [offs, shallow_offs+1000], n_freq_change, "shallow")

                wf_list += [wf_exp_static, wf_exp_shallow]
                if add_deep_wells:
                    wf_exp_static_deep = tu.static_waveform(
                        pos, freq, offs, label + " deep", solv_wghts=deep_weights)
                    wf_list += [wf_exp_static_deep]

        if add_reordering:
            wf_list += ror.generate_reorder_wfms(wf_exp_dual_species,
                                                 [1.0,1.5,2.0,2.5], [0], 100)

        wfs_load = WaveformSet(wf_list)
        wfs_load.write(wfm_path)

    if analyse_wfms:
        pot = calculate_potentials(trap_mom, wfs_load.get_waveform(2))
        plot_selection(pot)
        print(pot.find_wells(0, mode='precise'))
        print(pot.find_wells(100, mode='precise'))
    
    return wfs_load

if __name__ == "__main__":
    import datetime as dt
    default_freq = 1.1
    conveyor_offs = 1000
    n_load = 1001

    # Loading conveyor waveform
    wf_load_conveyor = tu.conveyor_waveform(
        [-1870, 0], [0.7, default_freq], [600, conveyor_offs],
        2*n_load, "Load -> exp", linspace_fn=zpspace)
    wp_load_conveyor = WavPotential(wf_load_conveyor)

    if True:
        wf_load_conveyor_rec = tu.conveyor_rec_waveform(
            [-1870, 0], [0.7, default_freq], [600, conveyor_offs],
            2*n_load, "Load -> exp", linspace_fn=zpspace)
        wp_lcr = WavPotential(wf_load_conveyor_rec)

    wp = wp_load_conveyor
    wp_old = WavPotential(WaveformSet(waveform_file="waveform_files/load_split_2Be1Ca_2017_02_21_v04.dwc.json").get_waveform(1))
    st()
    # new_wells = 
    
    now = dt.datetime.now()
    load_wfm_path = os.path.join("temp", "TEST_loading_" + now.strftime("%Y_%m_%d.%H_%M"))

    
