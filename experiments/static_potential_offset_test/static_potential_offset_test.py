#!/usr/bin/env python3

import sys
sys.path.append("../../")
from pytrans import *
import transport_utils as tu

import os
import csv
import copy as cp

# local_weights = {'r0':1e-6,
#                  'r0_u_weights':np.ones(30)*1e-4,
#                  'r0_u_ss':np.ones(30)*8,
#                  'r1':1e-6,'r2':1e-7}

# default_weights = {'r0':1e-9,
#                    'r0_u_weights':np.ones(30), # all electrodes uniform
#                    'r0_u_ss':np.ones(30)*default_elec_voltage,
#                    'r1':5e-5,'r2':0}

# local_potential_params={'energy_threshold':10*meV}

wf_path = os.path.join(os.pardir, os.pardir, "waveform_files", "static_potential_offsets_2017_05_24_v01.dwc.json")


def single_waveform(Gain_Error_Analysis=False, Gain_Error=1.0, Fault_Analysis=False, Faulty_Voltage=0.0, New_DEATH_Gain_Error=False):
    # w_desired = WavDesiredWells(0, 1.6, 70, "Single waveform test")
    #                             [np.array([1.6])*MHz],
    #                             [np.array([70])*meV],
                                
    #                             solver_weights=local_weights,
    #                             desired_potential_params=local_potential_params,                                
                                
    #                             desc="Single waveform test")

    wf_list = []
    wf = tu.transport_waveform([np.array([0])*um],
                                [np.array([1.6])*MHz],
                                [np.array([70])*meV],

        
    if New_DEATH_Gain_Error:
        for i in range(16):
            wf.samples[i][0] *= 1.027
    if Gain_Error_Analysis:
        for i in range(len(wf.samples)):
            wf.samples[i][0] *= Gain_Error
    if Fault_Analysis:
        wf.samples[physical_electrode_transform[[10]]] = np.array([[Faulty_Voltage]]).T
    # Un-comment the next line to simulate electrode 22 not working:
    #wf.samples[physical_electrode_transform[[22]]] = np.array([[0.0]]).T
    wf_list.append(wf)

    # electrode_offsets = [[-10],[-10],[-10],[10],[10],[5],[1.5],[0.5],[1.5],[5],[10],[10],[-10],[-10],[-10],
    #                      [-10],[-10],[-10],[10],[10],[5],[1.5],[0.5],[1.5],[5],[10],[10],[-10],[-10],[-10]]

    # electrode_combos = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],
    #                     [15],[16],[17],[18],[19],[20],[21],[22],[23],[24],[25],[26],[27],[28],[29]]
    # electrode_offsets = [[-0.15], [+0.15]]
    # electrode_combos = [[7],[7]]

    # For analysis: reading files generated by analysis_160605.py
    electrode_offsets = []
    with open('spot_offset_inputs.csv') as spot_off_in:
        for offset in csv.reader(spot_off_in, delimiter=' '):
            offset_list = []
            for of in offset:
                if of != '':
                    offset_list.append(float(of))
            electrode_offsets.append(offset_list)
    
    electrode_combos = []
    with open('spot_combo_inputs.csv') as spot_com_in:
        for combo in csv.reader(spot_com_in, delimiter=' '):
            combo_list = []
            for co in combo:
                if co != '':
                    combo_list.append(int(co))
            electrode_combos.append(combo_list)

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

def analyze_waveform(num_of_cases, plotting=True, printing=True, writing=False):
    if plotting:
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

    starting_f = None
    
    for i in range(num_of_cases+1): # +1 here is to account for the first potential for which no voltages have been altered
        pot = calculate_potentials(trap_mom, wfs.get_waveform(i+1))
        if plotting:
            pot.plot_one_wfm(0, ax)
        calc_freq = ( pot.find_wells(0, mode='quick'), pot.find_wells(0, mode='precise') )
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

    if plotting:
        plt.show()

    if printing:
        # print(calc_freq_list)
        for fq, fp, dfq, dfp, dsfqp  in zip(f_desired_q[1:], f_desired_p[1:], delta_f_q, delta_f_p, discrep_f_qp[1:]):
            print("f desir q: {:.3f} kHz, f desir p: {:.3f} kHz, delta f q: {:.3f} kHz, delta f p: {:.3f} kHz, discrep f p - q: {:.3f}".format(fq/1e3,fp[0]/1e3,dfq/1e3,dfp[0]/1e3,dsfqp[0]/1e3))

    if writing:
        os.system('touch delta_f_p.csv')
        with open('delta_f_p.csv', 'w', newline='') as dfp:
            csv.writer(dfp).writerows(delta_f_p)

        os.system('touch f_p.csv')
        with open('f_p.csv', 'w', newline='') as fp:
            csv.writer(fp).writerows(f_desired_p)

if __name__ == "__main__":
    # load_to_exp()

    with open('analysis_options.txt') as anop:
        GEA = bool(int(anop.readline()))
        GE = float(anop.readline())
        FA = bool(int(anop.readline()))
        FV = float(anop.readline())
        NDGE = bool(int(anop.readline()))

    case_num = single_waveform(Gain_Error_Analysis=GEA, Gain_Error=GE, Fault_Analysis=FA, Faulty_Voltage=FV, New_DEATH_Gain_Error=NDGE)
    analyze_waveform(case_num, plotting=False, printing=False, writing=True)
