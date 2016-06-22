from pytrans import *

def calculate_potentials(moments, waveform,
                         real_electrode_idxes=physical_electrode_transform,
                         ):
    """ 
    Multiplies the moments matrix by the waveform matrix (with suitable truncation based on real_electrode_idxes parameter)
    moments: Moments class containing potential data
    waveform: Waveform class containing the voltage samples array
    """
    mom_trunc = moments.potentials[:,:len(real_electrode_idxes)]
    waveform_trunc = waveform.samples[real_electrode_idxes,:]
    
    return WavPotential(np.dot(mom_trunc, waveform_trunc), moments.transport_axis, 39.962591)

def plot_td_voltages(waveform, electrodes_to_use=None, real_electrodes=physical_electrode_transform):
    """ Plot time-dependent voltages of a waveform w.r.t. electrodes as"""
    td_wfms = waveform.samples.T
    if electrodes_to_use:
        td_wfms = td_wfms[electrodes_to_use]
        leg = tuple(str(k+1) for k in electrodes_to_use)
    else:
        leg = tuple(str(k+1) for k in range(real_electrodes))
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(td_wfms)
    ax.legend(leg)
    plt.show()
    
if __name__ == "__main__":
    stationary_comparison_with_old = False
    check_splitting_waveform = False
    check_loading_waveform = False
    check_conveyor_waveform = False
    check_trap_modes = True
    
    mom = Moments()

    if stationary_comparison_with_old:
        wf = WaveformSet(waveform_file='waveform_files/Ca_trans_load_open_Ca_Be_Transport_scan_freq_and_offset_pos_0_um.dwc.json')

        wf_load_54 = wf.get_waveform('wav54')
        pot_load_54 = calculate_potentials(mom, wf_load_54)

        wf_load_62 = wf.get_waveform('wav62')
        pot_load_62 = calculate_potentials(mom, wf_load_62)

        wf_load_104 = wf.get_waveform('wav104')
        pot_load_104 = calculate_potentials(mom, wf_load_104)    

        wf2 = WaveformSet(waveform_file='waveform_files/loading_and_constant_settings_Ts_620_2016_04_25_v01.dwc.json')

        wfms = (15, 24, 132) # or (17, 
        wfms_new = tuple(wf2.get_waveform(k) for k in wfms)
        pot_loads = tuple(calculate_potentials(mom, wf) for wf in wfms_new)

        def well_search():
            indices = []
            trap_freqs = []
            wfms = np.arange(500)
            for k in wfms:
                ind, _, tf = pot_load.find_wells(k)
                try:
                    indices.append(ind[np.argmax(tf)])
                except IndexError:
                    st()
                trap_freqs.append(tf.max())

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax = fig.add_subplot(111)
            st()
            ax.plot(np.array(wfms, dtype='float64'),
                    np.array(indices,dtype='float64'),
                    np.array(trap_freqs,dtype='float64'))
            plt.show()

        # well_search()
        test_wf = wf.get_waveform(8)
        plot_td_voltages(test_wf)
        # pot_load.find_wells(0)

        axa = pot_load_54.plot_one_wfm(0)
        pot_load_62.plot_one_wfm(0, axa)
        pot_load_104.plot_one_wfm(0, axa)    
        # pot_load.plot_one_wfm(-1, axa)

        for pl in pot_loads:
            pl.plot_one_wfm(0, axa)
        #pot_load2.plot_one_wfm(0, axa)
    #    pot_load2.plot_one_wfm(-1, axa)    

        # plt.plot(pot_load.potentials[:,990])
        # plt.show()
        # pot_load.plot()
        plt.show()
    
    if check_splitting_waveform:
        wf = WaveformSet(waveform_file='waveform_files/splitting_zone_Ts_620_vn_2016_04_14_v03.dwc.json')

        wf_all_sections = wf.get_waveform('wav8')
        pot_all_sections = calculate_potentials(mom, wf_all_sections)

        pot_all_sections.plot()
        plt.show()
    
    if check_loading_waveform:
        wf = WaveformSet(waveform_file='waveform_files/loading_and_constant_settings_Ts_620_2016_04_25_v01.dwc.json')

        wf_load_to_exp = wf.get_waveform('wav1')
        pot_load_to_exp = calculate_potentials(mom, wf_load_to_exp)

        pot_load_to_exp.plot()
        plt.show()
        
    if check_conveyor_waveform:
        wf = WaveformSet(waveform_file='waveform_files/loading_conveyor_Ts620_2016_04_26_v01.dwc.json')

        wf_load_to_exp = wf.get_waveform('wav1')
        pot_load_to_exp = calculate_potentials(mom, wf_load_to_exp)

        pot_load_to_exp.plot()
        plt.show()

    if check_trap_modes:
        wf_all = WaveformSet(waveform_file='waveform_files/loading_and_constant_settings_Ts_620_2016_04_25_v01.dwc.json')
        wf_to_try = 'wav133'
        wf = wf_all.get_waveform(wf_to_try)

        pot = calculate_potentials(mom, wf)
        # st()
        print(pot.find_wells(2, 'quick'))
        print(pot.find_wells(2, 'precise'))
        # pot.plot()
        # plt.show()
