#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

def single_waveform():
    wf_path = os.path.join(os.pardir, "waveform_files", "single_test_waveform.dwc.json")
    w_desired = WavDesiredWells([np.array([0])*um],
                                [np.array([1.5])*MHz],
                                [np.array([800])*meV],
                                solver_weights={'energy_threshold':200*meV,
                                                'r0_u_weights':np.ones(30)*3e-2,
                                                'r0_u_ss':np.ones(30)*8},
                                desc="Single waveform test")
    wf_list = []
    wf = Waveform(w_desired)
    wf_list.append(wf)

    # physical_electrode_transform = [0,4,8,2,  6,10,14,18,  22,26,30,16,  20,24,13,
    #                                 1,5,9,3,  7,11,15,19,  23,27,31,17,  21,25,29]

    # desired_electrode_shifts = [10,10,10,10, 10,5,1.5,0.5, 1.5,5,10,10, 10,10,10,
    #                             10,10,10,10, 10,5,1.5,0.5, 1.5,5,10,10, 10,10,10]

    # The three end electrodes have little effect on the curvature of
    # the potential at the center of the trap, so we combine them to
    # obtain a larger effect; multiple_electrode_transform keeps track
    # of which triples we will apply together.

    multiple_electrode_transform = [0,4,8, 20,24,13,
                                    1,5,9, 21,25,29]

    multiple_electrode_shifts = [10,10,10, 10,10,10,
                                 10,10,10, 10,10,10]

    count = 0
    for met, mes in zip(multiple_electrode_transform, multiple_electrode_shifts):
        if count%3 ==0:
            wf = Waveform(w_desired)
        count = count + 1
        wf.samples[met,0] = wf.samples[met,0] + mes
        if count%3 == 0:
            wf_list.append(wf)

    # The rest of the center electrodes are still altered
    # individually:

    individual_electrode_transform = [2, 6,10,14,18, 22,26,30,16,
                                      3, 7,11,15,19, 23,27,31,17]

    individual_electrode_shifts = [10, 10,5,1.5,0.5,1.5, 5,10,10,
                                   10, 10,5,1.5,0.5,1.5, 5,10,10]
    
    # used_chans = list(range(12)) + list(range(13,28)) + list(range(29,32))
    # for i in physical_electrode_transform:
    for iet, ies in zip(individual_electrode_transform, individual_electrode_shifts):
        wf = Waveform(w_desired)
        wf.samples[iet,0] = wf.samples[iet,0] + ies
        wf_list.append(wf)

    # reorder electrode changes:
    wf_list_reordered = []
    for i in range(2):
        wf_list_reordered.append(wf_list[i])
    for j in range(5,14):
        wf_list_reordered.append(wf_list[j])
    for k in range(2,4):
        wf_list_reordered.append(wf_list[k])
    for m in range(14,23):
        wf_list_reordered.append(wf_list[m])
    wf_list_reordered.append(wf_list[4])
    

    wfs = WaveformSet(wf_list_reordered)
    wfs.write(wf_path)

    return len(wf_list)

def analyze_waveform(case_number):
    wf_path = os.path.join(os.pardir, "waveform_files", "single_test_waveform.dwc.json")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    wfs = WaveformSet(waveform_file=wf_path)

    pot = calculate_potentials(trap_mom, wfs.get_waveform(1))
    calc_freq_list = [ ( pot.find_wells(0, mode='quick'), pot.find_wells(0, mode='precise') ) ]

    f_desired_q = []
    f_desired_p = []    
    ind_des_q = []
    ind_des_p = []
    off_des_q = []
    off_des_p = []

    for i in range(case_number):
        pot = calculate_potentials(trap_mom, wfs.get_waveform(i+1))
        pot.plot_one_wfm(0, ax)
        calc_freq = ( pot.find_wells(0, mode='quick'), pot.find_wells(0, mode='precise') )
        calc_freq_list.append(calc_freq)

        indices = tuple(k['min_indices'] for k in calc_freq)
        offsets = tuple(k['offsets'] for k in calc_freq)
        freqs = tuple(k['freqs'] for k in calc_freq)

        for index_q, index_p, offset_q, offset_p, freq_q, freq_p in zip(indices[0], indices[1], offsets[0], offsets[1], freqs[0], freqs[1]):
            
            if freq_q > 1e6: # a statement that we would like to be true
                           # is used in this case since any comparison
                           # with a 'nan' value will return False
                           # (there are some minima for wich the
                           # frequency returns a 'nan' value, but we
                           # do not need those values for the purposes
                           # of this code)
                f_desired_q.append(freq_q)
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
    for fq, fp, dfq, dfp, dsfqp  in zip(f_desired_q, f_desired_p, delta_f_q, delta_f_p, discrep_f_qp):
        print("f desir q: {:.3f} kHz, f desir p: {:.3f} kHz, delta f q: {:.3f} kHz, delta f p: {:.3f} kHz, discrep f p - q: {:.3f}".format(fq/1e3,fp[0]/1e3,dfq/1e3,dfp[0]/1e3,dsfqp[0]/1e3))

    st()

if __name__ == "__main__":
    # load_to_exp()
    num_of_cases = single_waveform()
    analyze_waveform(num_of_cases)
