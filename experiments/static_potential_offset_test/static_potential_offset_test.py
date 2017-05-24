#!/usr/bin/env python3

import sys
sys.path.append("../../")
from pytrans import *
import transport_utils as tu

import copy as cp

# local_weights = {'r0':1e-6,
#                  'r0_u_weights':np.ones(30)*1e-4,
#                  'r0_u_ss':np.ones(30)*8,
#                  'r1':1e-6,'r2':1e-7}

# local_potential_params={'energy_threshold':10*meV}

wf_path = os.path.join(os.pardir, os.pardir, "waveform_files", "static_potential_offsets_2017_05_24_v01.dwc.json")

def single_waveform():
    # w_desired = WavDesiredWells([np.array([0])*um],
    #                             [np.array([1.6])*MHz],
    #                             [np.array([70])*meV],
                                
    #                             solver_weights=local_weights,
    #                             desired_potential_params=local_potential_params,                                
                                
    #                             desc="Single waveform test")
    wf_list = []

    solv_wghts = cp.copy(tu.default_weights)
    solv_wghts['r0_u_ss'] = np.zeros(30)
    
    wf = tu.static_waveform(0, 1.8, 1600, "Single waveform test", solv_wghts)
    
    # wf = Waveform(w_desired)
    wf_list.append(wf)

    # electrode_offsets = [[-10],[-10],[-10],[10],[10],[5],[1.5],[0.5],[1.5],[5],[10],[10],[-10],[-10],[-10],
    #                      [-10],[-10],[-10],[10],[10],[5],[1.5],[0.5],[1.5],[5],[10],[10],[-10],[-10],[-10]]

    # electrode_combos = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],
    #                     [15],[16],[17],[18],[19],[20],[21],[22],[23],[24],[25],[26],[27],[28],[29]]
    # electrode_offsets = [[-0.15], [+0.15]]
    # electrode_combos = [[7],[7]]

    big_shift = 5
    
    electrode_offsets = [[-0.4], [+0.4], [-0.4], [0.4],
                         [-1], [1], [-1], [1], 
                         [-4],[4],[-4],[4],
                         [-big_shift],[big_shift],[-big_shift],[big_shift],
                         [-0.4], [+0.4], [-0.4], [0.4],
                         [-1], [1], [-1], [1], 
                         [-4],[4],[-4],[4],
                         [-big_shift],[big_shift],[-big_shift],[big_shift],
                         # end electrodes, need strong voltages
                         [-big_shift,-big_shift,-big_shift],[big_shift,big_shift,big_shift],
                         [-big_shift,-big_shift,-big_shift],[big_shift,big_shift,big_shift],
                         [-big_shift,-big_shift,-big_shift],[big_shift,big_shift,big_shift],
                         [-big_shift,-big_shift,-big_shift],[big_shift,big_shift,big_shift]]
    
    electrode_combos = [[6],[6], [8],[8],
                        [5],[5], [9],[9],
                        [4],[4],[10],[10],
                        [3],[3],[11],[11],
                        [21],[21], [23],[23],
                        [20],[20], [24],[24],
                        [19],[19],[25],[25],
                        [18],[18],[26],[26],
                        # end electrodes, need strong voltages
                        [0,1,2],[0,1,2],
                        [12,13,14],[12,13,14],
                        [15,16,17],[15,16,17],
                        [27,28,29],[27,28,29]]

    for ec, eo in zip(electrode_combos, electrode_offsets):
        voltages = wf.samples[physical_electrode_transform[ec]]
        st()
        # print("Voltages for elec " + str(ec) + " are " + str(voltages))

    for ec, eo in zip(electrode_combos, electrode_offsets):
        assert len(ec) == len(eo), "Different number of electrodes and offsets requested!"
        wf2 = cp.deepcopy(wf)
        wf2.set_new_uid()
        wf2.desc = ""
        for ec_l, eo_l in zip(ec, eo):
            try:
                v_str = ""
                elec_str = ""
                for ec_ll, eo_ll in zip(ec_l, eo_l):
                    v_str += "{:d}".format(ec_ll)
                    elec_str += "{:.2f}".format(eo_ll)
                wf2.desc += v_str + " V offset on Elec " + elec_str
            except TypeError:
                wf2.desc += "{:.2f} V offset on Elec {:d}, ".format(eo_l, ec_l)
        wf2.samples[physical_electrode_transform[ec]] += np.array([eo]).T
        if (wf2.voltage_limits_exceeded()):
            print("Error in electrode " + str(ec))
        wf_list.append(wf2)
                
    wfs = WaveformSet(wf_list)
    wfs.write(wf_path)

    return len(electrode_combos)

def analyze_waveform(num_of_cases):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    wfs = WaveformSet(waveform_file=wf_path)

    pot = WavPotential(wfs.get_waveform(1))
    # pot = calculate_potentials(trap_mom, wfs.get_waveform(1))
    # st()
    # calc_freq_list = [ ( pot.find_wells(0, mode='quick'), pot.find_wells(0, mode='precise') ) ]
    calc_freq_list = [[ pot.find_wells(0, mode='quick', roi_width=1e-3),
                       pot.find_wells(0, mode='precise', roi_width=1e-3)]]

    f_desired_q = []
    f_desired_p = []
    ind_des_q = []
    ind_des_p = []
    off_des_q = []
    off_des_p = []

    starting_f = None
    
    for i in range(num_of_cases):
        # pot = calculate_potentials(trap_mom, wfs.get_waveform(i+1))
        pot = WavPotential(wfs.get_waveform(i+1)) # +1 here to account for 1st waveform, which has not been altered
        pot.plot_one_wfm(0, ax)
        calc_freq = [ pot.find_wells(0, mode='quick', roi_width=1e-3),
                       pot.find_wells(0, mode='precise', roi_width=1e-3) ]
        calc_freq_list.append(calc_freq)

        indices = tuple(k['min_indices'] for k in calc_freq)
        offsets = tuple(k['offsets'] for k in calc_freq)
        freqs = tuple(k['freqs'] for k in calc_freq)
        locs = tuple(k['locs'] for k in calc_freq)

        for index_q, index_p, offset_q, offset_p, freq_q, freq_p, locs_q, locs_p in zip(indices[0], indices[1], offsets[0], offsets[1], freqs[0], freqs[1], locs[0], locs[1]):
            if -20*um<locs_q<20*um:
            # if freq_q > 1e6: # a statement that we would like to be true
            #                # is used in this case since any comparison
            #                # with a 'nan' value will return False
            #                # (there are some minima for wich the
            #                # frequency returns a 'nan' value, but we
            #                # do not need those values for the purposes
            #                # of this code)
                f_desired_q.append(freq_q)
                if not starting_f:
                    starting_f = freq_q
                if (freq_q - starting_f) < -100*kHz:
                    st()
                f_desired_p.append(freq_p)                
                ind_des_q.append(index_q)
                ind_des_p.append(index_p)                
                off_des_q.append(offset_q)
                off_des_p.append(offset_p)

    # here delta_f is defined as: altered frequency - unaltered frequency
    delta_f_q = tuple(k - f_desired_q[0] for k in f_desired_q[1:])
    delta_f_p = tuple(k - f_desired_p[0] for k in f_desired_p[1:])
    # checking if the discrepancy between frequency desired quick and frequency desired precise is consistent or if there are any trends
    discrep_f_qp = []
    for j in range(len(f_desired_q)):
        discrep_f_qp.append(f_desired_p[j] - f_desired_q[j])

    plt.show()
    # print(calc_freq_list)
    print('q: quick, p: precise')
    for fq, fp, dfq, dfp, dsfqp  in zip(f_desired_q[1:], f_desired_p[1:], delta_f_q, delta_f_p, discrep_f_qp[1:]):
        # print("f desir q: {:.3f} kHz, f desir p: {:.3f} kHz, delta f q: {:.3f} kHz, delta f p: {:.3f} kHz, discrep f p - q: {:.3f}".format(fq/1e3,fp[0]/1e3,dfq/1e3,dfp[0]/1e3,dsfqp[0]/1e3))
        print("f desir q: {:.3f} kHz, f desir p: {:.3f} kHz, delta f q: {:.3f} kHz, delta f p: {:.3f} kHz, discrep f p - q: {:.3f}".format(fq/1e3,fp/1e3,dfq/1e3,dfp/1e3,dsfqp/1e3))

if __name__ == "__main__":
    # load_to_exp()
    case_num = single_waveform()
    analyze_waveform(case_num)
