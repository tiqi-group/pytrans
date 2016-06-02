#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

def single_waveform():
    wf_path = os.path.join(os.pardir, "waveform_files", "single_test_waveform.dwc.json")
<<<<<<< HEAD
    w_desired = WavDesiredWells(np.array([0])*um,
                                np.array([2.0])*MHz,
                                np.array([800])*meV,
=======
    w_desired = WavDesiredWells(np.array([[0]])*um,
                                np.array([[1.5]])*MHz,
                                np.array([[800]])*meV,
>>>>>>> 268a9b7334d312ccd80412e9ec00747567c693bc
                                solver_weights={'energy_threshold':200*meV,
                                                'r0_u_weights':np.ones(30)*3e-2,
                                                'r0_u_ss':np.ones(30)*8},
                                desc="Single waveform test")
    wf_list = []
    wf = Waveform(w_desired)
    wf_list.append(wf)
    used_chans = list(range(0,12)) + list(range(13,28)) + list(range(29,32))
    for i in used_chans:
        wf = Waveform(w_desired)
        wf.samples[i,0] = wf.samples[i,0] + 1
        wf_list.append(wf)

    wfs = WaveformSet(wf_list)
    wfs.write(wf_path)

def analyze_waveform():
    wf_path = os.path.join(os.pardir, "waveform_files", "single_test_waveform.dwc.json")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    wfs = WaveformSet(waveform_file=wf_path)
    calc_freq_list = []
    
    pot = calculate_potentials(trap_mom, wfs.get_waveform(1))
    calc_freq = pot.find_wells(0, mode='precise')
    calc_freq_list.append(calc_freq)

    f_desired = []

    # here delta_f is defined as: altered frequency - unaltered frequency
    delta_f = []
    
    for i in range(0,31):
        pot = calculate_potentials(trap_mom, wfs.get_waveform(i+1))
        pot.plot_one_wfm(0, ax)
        calc_freq = pot.find_wells(0, mode='precise')
        calc_freq_list.append(calc_freq)

        minima = calc_freq_list[i][0]

        for j in range(0, minima.size):
            if calc_freq_list[i][1][j] < 0.5:
                f_desired.append(calc_freq_list[i][2][j][0])
                if i > 0:
                    delta_f.append(f_desired[i] - f_desired[0])
        
    plt.show()
    # print(calc_freq_list)
    print(delta_f)
    print(f_desired)
    st()

if __name__ == "__main__":
    # load_to_exp()
    single_waveform()
    analyze_waveform()
