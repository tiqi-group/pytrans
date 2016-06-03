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

    # Same as in pytrans.py
    physical_electrode_transform = [0,4,8,2,  6,10,14,18,  22,26,30,16,  20,24,13,
                                    1,5,9,3,  7,11,15,19,  23,27,31,17,  21,25,29]

    # Here you enter the voltage shifts you would like to apply
    desired_electrode_shifts = [10,10,10,10, 10,5,1.5,0.5, 1.5,5,10,10, 10,10,10,
                                10,10,10,10, 10,5,1.5,0.5, 1.5,5,10,10, 10,10,10]

    # Here you enter the combinations of electrode shifts you would
    # like to apply simultaneously or individually. e.g. if there were
    # 8 electrodes, and you wanted to apply the first and last at the
    # same time, the second and before last at the same time, and the
    # rest individually, you could write electrode_combos =
    # [1,2,3,4,5,6,2,1]. In this example we had 6 cases. Note that
    # data will be output in the order that the numdering of the cases
    # is written in electrode_combos. Numbering starts at 1, and don't
    # skip any numbers, since this will result in saving additional
    # trivial cases for the skipped numbers.

    electrode_combos = [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,
                        16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30]

    electrode_combos = [[1,2],[5],[7]]
    electrode_offsets = [[0.5,1.76],[5],[-4]]

    import copy as cp

    for ec, eo in zip(electrode_combos, electrode_offsets):
        wf2 = cp.deepcopy(wf)
        # wf2.
        wf2.samples[ec] += np.array(eo)
    
    # find the number of cases:
    ec_sort = electrode_combos
    ec_sort.sort
    cases = ec_sort[29]

    for case in range(1, cases+1):
        wf = Waveform(w_desired)
        for vset in range(11): # create a set of voltage potentials in 10 steps going from positive to negative voltage (or vice versa depending on the initial signs of the assigned voltages)
            # This next step is not the most efficient since we only really need the voltages of the electrodes in this set
            incremental_electrode_shifts = tuple(des - des*vset/5 for des in desired_electrode_shifts[:])
            for pet,ies,ec in zip(physical_electrode_transform, incremental_electrode_shifts, electrode_combos):
                if ec == case:
                    wf.samples[pet,0] = wf.samples[pet,0] + ies
        wf_list.append(wf)

            
    wfs = WaveformSet(wf_list)
    wfs.write(wf_path)

    return cases

def analyze_waveform(num_of_cases):
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

    for i in range(num_of_cases+1): # +1 here is to account for the first potential for which no voltages have been altered
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
    case_num = single_waveform()
    analyze_waveform(case_num)
