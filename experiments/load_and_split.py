#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from reorder import *

# Loading conveyor stuff
import loading_conveyor as lc

# Splitting (need to refactor soon - there's a lot of unneeded stuff in splitting.py!)
import splitting as sp

def split_waveform(plot_splits=False):
    electrode_subset = [3,4,5,6,7,18,19,20,21,22] # left splitting group
    z_centre = -422.5*um # centre of the central splitting electrode moment
    polyfit_range = 200*um

    polys = sp.generate_interp_polys(trap_mom.transport_axis,
                                    trap_mom.potentials[:, electrode_subset],
                                    z_centre, polyfit_range)

    # Initial waveform, after transport to the splitting location
    wf_split = lc.static_waveform(-422.5, 1.356, -1475, '')
    
    # Data format is (alpha, slope, points from prev. state to this one)
    split_params = [# (1.5e7, None, 500, np.linspace),
        # (1e6, None, 500, np.linspace),
                    #(0, 0.16, 500, lambda a,b,n: erfspace(a,b,n,1.5)),
        (1e6, None, 200, np.linspace),
        (0, None, 50, np.linspace),
        # (-3e6, None, 500, np.linspace),
        (-5e6, None, 50, np.linspace),
        (-1e7, None, 50, np.linspace),
        (-2e7, None, 100, np.linspace),
        (-3e7, None, 100, np.linspace),
        (-4e7, None, 100, np.linspace),
        (-5e7, None, 150, np.linspace),
        (-6e7, None, 300, np.linspace)]
    
    last_death_voltages = wf_split.samples
    full_wfm_voltages = last_death_voltages.copy()

    debug_splitting_parts = False
    # Prepare full voltage array
    for (alpha, slope_offset, npts, linspace_fn) in split_params:
        elec_voltage_set,_,_ = sp.solve_poly_ab(polys, alpha,
                                            slope_offset=slope_offset, dc_offset=None)
        new_death_voltages = last_death_voltages.copy()
        new_death_voltages[physical_electrode_transform[electrode_subset]] = elec_voltage_set

        # Ramp from old to new voltage set
        ramped_voltages = vlinspace(last_death_voltages, new_death_voltages, npts, linspace_fn)[:,1:]
        full_wfm_voltages = np.hstack([full_wfm_voltages, ramped_voltages])
        last_death_voltages = new_death_voltages

        if debug_splitting_parts:
            new_wf = Waveform("", 0, "", ramped_voltages)        
            asdf = WavPotential(new_wf)
            asdf.plot_range_of_wfms(20)
            plt.show()

    # Final waveform, extends separation by 150um either way and goes to default well settings
    # (which may need adjustment!)
    wf_finish_split = lc.transport_waveform_multiple(
        [[-605, -755],[-240, -90]],
        [[2.7,1.3],[2.7,1.3]],
        [[-538, 960],[-538, 960]],
        500,
        "")

    full_wfm_voltages = np.hstack([full_wfm_voltages, wf_finish_split.samples[:,1:]])
            
    splitting_wf = Waveform("split test", 0, "", full_wfm_voltages)
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

    return splitting_wf

def load_and_split(add_reordering=True, analyse_wfms=False):
    wf_path = os.path.join(os.pardir, "waveform_files", "load_split_2016_06_22_v01.dwc.json")
    # If file exists already, just load it to save time
    try:
        # raise FileNotFoundError # uncomment to always regenerate file for debugging
        wfs_load_and_split = WaveformSet(waveform_file=wf_path)
        print("Loaded waveform ",wf_path)
    except FileNotFoundError:
        print("Generating waveform ",wf_path)
        # use existing loading conveyor file to save time - need to regenerate if not available
        wf_load_path = os.path.join(os.pardir, "waveform_files", "loading_2016_06_21_v01.dwc.json")
        wfs_load = WaveformSet(waveform_file=wf_load_path)

        n_transport = 1000

        # Transport from loading to splitting zone        
        wf_load_split = lc.transport_waveform(
            [0, -422.5], [1.3, 1.356], [960, -1475],
            n_transport, "Exp -> left split")
        wfs_load_and_split = wfs_load
        wfs_load_and_split.waveforms.append(wf_load_split)        
        
        wf_split = split_waveform()
        wfs_load_and_split.waveforms.append(wf_split)
        wfs_load_and_split.write(wf_path)

    # Create a single testing waveform
    add_testing_waveform = True
    if add_testing_waveform:
        test_waveform_present = wfs_load_and_split.get_waveform(-1).desc == "trans + split, then reverse"
        if test_waveform_present:        
            exp_to_split_wfm = wfs_load_and_split.get_waveform(-3)
            end_wfm = wfs_load_and_split.get_waveform(-2)
        else:
            exp_to_split_wfm = wfs_load_and_split.get_waveform(-2)
            end_wfm = wfs_load_and_split.get_waveform(-1)

        trans_split_forward = np.hstack([exp_to_split_wfm.samples, end_wfm.samples[:,:-500]])
        trans_split_for_rev = np.hstack([trans_split_forward, np.fliplr(trans_split_forward)])
        wf_trans_split_for_rev = Waveform("trans + split, then reverse", 0, "", trans_split_for_rev)
        wf_trans_split_for_rev.set_new_uid()

        if test_waveform_present:
            wfs_load_and_split.waveforms[-1] = wf_trans_split_for_rev
        else:
            wfs_load_and_split.waveforms.append(wf_trans_split_for_rev)

        # pot = WavPotential(wf_trans_split_for_rev)
        # pot.animate_wfm()
        wfs_load_and_split.write(wf_path)

if __name__ == "__main__":
    load_and_split()
