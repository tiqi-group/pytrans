# Data analysis script for static potential offset experiment on June 5th 2016.

import sys
sys.path.append("../../")
from pytrans import physical_electrode_transform
import os
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.tight_layout as tlout
from scipy.optimize import curve_fit
from textwrap import wrap
import pdb
st = pdb.set_trace

# Make sure to end paths with a '/':
path_to_sim = './' # path to the pytrans simulation static_potential_offset_test.py
path_to_data = '../../../../experiments/160605/' # wherever you put the lab data
file_to_plot = 'DC GS Tickle for cal z Ca_0_f z motional Ca_plot_0.csv'
# Need to make these directories if they don't already exist:
path_to_plot_data = './plots/data/' # wherever you want to save data plot figures
path_to_plot_asys = './plots/analysis/' # wherever you want to save analysis plot figures

ext = '.pdf' # whatever file type you want to save your plot figures as
plot_viewer = 'evince'
py_command = 'python3.4' # the command to run python in shell for your system

# what you want to plot on the x and y axes -- needs to be a string contained
# in the fist row of the data file (the first row of the data file contains
# the descriptions of the data):
xaxis = 'x' # ' y' ' time [s]'
yaxis = ' Ca+ signal (s)' # 'DDS err.' ' BkgCorr counts' ' Timing err.'


# Contents of the table we filled in on the day of the experiments:
timestamps = (183353, 183622, 183731, 184523, 184806, 185005, 185412, 185602, 185807, 185953, 190115, 190241, 190423, 190602, 190831, 191224, 191501, 191748, 191925, 192056, 192307, 192442, 193508, 193906, 194209, 194402, 194601, 194723, 194956, 195114, 195330, 195633, 195908, 200058, 200303, 200709, 202753, 202954, 203127, 203245, 203411, 203531, 203655, 203822)
electrodes = (7,7,7,6,6,6,5,5,8,8,9,9,4,4,10,10,3,3,11,11,21,21,23,23,20,20,24,24,19,19,25,25,18,18,26,26,[0,1,2],[0,1,2],[12,13,14],[12,13,14],[15,16,17],[15,16,17],[27,28,29],[27,28,29])
offsets = (0., -0.15, 0.15, 0., -0.4, 0.4, -1, 1, -0.4, 0.4, -1, 1, -4, 4, -4, 4, -8.5, 8.5, -8.5, 8.5, -0.4, 0.4, -0.4, 0.4, -1, 1, -1, 1, -4, 4, -4, 4, -8.5, 8.5, -8.5, 8.5, -8.5, 8.5, -8.5, 8.5, -8.5, 8.5, -8.5, 8.5)
expected_freqs = (1598, 1610, 1586, 1598, 1589, 1606, 1589, 1607, 1589, 1607, 1589, 1607, 1589, 1607, 1589, 1607, 1592, 1604, 1592, 1604, 1586, 1610, 1586, 1610, 1589, 1607, 1589, 1607, 1589.5, 1606.5, 1590, 1606, 1592, 1604, 1592, 1604, -1, -1, -1, -1, -1, -1, -1, -1) # kHz
meas_freqs = (1641, 1655, 1629, 1642, 1627, 1656, 1634, 1649, 1630, 1652, 1632, 1650, 1635, 1649, 1642, 1642, 1638, 1646, 1637, 1647, 1630, 1653, 1627, 1656, 1633, 1651, 1634, 1651, 1634, 1651, 1635, 1650, 1638, 1648, 1637, 1648, 1642, 1643, 1641, 1643, 1642, 1643, 1642, 1643) # kHz
accurs = ('', '', '', '', 'good', 'good', 'good', 'decent', 'good', 'decent', 'decent', 'decent', 'decent', 'decent', 'decent', 'decent', 'decent', 'sosolala', 'good', 'sosolala', 'decent', 'sosolala', 'decent', 'decent', 'decent', 'sosolala', 'decent', 'decent', 'decent', 'sosolala', 'good', 'sosolala', 'decent', 'sosolala', 'decent', 'sosolala', 'decent', 'decent', 'decent', 'decent', 'decent', 'good', 'decent', 'decent')
notes = ('', '', '', '', '', '', '', '', '', '', '', '', '', '', 'NO EFFECT', 'NO EFFECT', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '')

assert len(timestamps)==len(electrodes)==len(offsets)==len(expected_freqs)==len(meas_freqs)==len(accurs)==len(notes), 'check yourself before you wreck yourself'

# To use for fitting data below:
def Lorentzian(x, fwhm, scale, x0, y0):
        return y0 - (scale / ((x - x0)**2 + (fwhm/2)**2))

def Analyze(Plot_Data=True, Shell_Out=True, Open_Plots=True, New_DEATH_Gain_Error=False, Gain_Error_Analysis=False, Gain_Error=1.0, Fault_Analysis=False, Faulty_Voltage=0.0):
    ########## Reading Data ############
    all_data_dict = {}
    all_fit_chars = {}
    for ts, el, of, ef, mf, ac, nt in zip(timestamps, electrodes, offsets, expected_freqs, meas_freqs, accurs, notes):
        path_to_data_file = path_to_data + '20160605_' + str(ts) + ' DC GS Tickle for cal z Ca/'
        with open(path_to_data_file + file_to_plot, newline='') as data:
            read_data_dict = csv.DictReader(data, delimiter=',')
            data_dict = {}
            column_list_declare = False
            for row in read_data_dict:
                if not column_list_declare:
                    column_list = [[] for i in row.keys()]
                    column_list_declare = True
                for key, index in zip(row.keys(), range( len(row.keys()) ) ):
                    column_list[index].append(float(row[key])) # column list is a list of lists containing floats (the data)
            del column_list_declare
            for key, index in zip(row.keys(), range( len(row.keys()) ) ):
                data_dict[key] = np.array(column_list[index]) # data_dict is a dictionary of numpy arrays
            del column_list
            
        with open(path_to_data_file + 'fit.txt') as fit:
            fit_params = fit.read()
            fp = []
            fit.seek(0)
            for i in range(5): # the range is determined by the size of the file fit.txt
                line = fit.readline()
                if i > 0:
                    fp.append( float(''.join(c for c in line if c.isdigit() or c =='.' or c == '-')) )
        
        # record data for all experiments into one object:
        # all_data_dict. all_data_dict is a dictionary whose keys are the
        # timestamps of the experiments. For each timestamp key, it contains a
        # dictionary of numpy arrays whose keys are the data type (given by the
        # first row of the csv data file), and whose numpy arrays are 1D and
        # contain the corresponding data.
        all_data_dict[str(ts)] = data_dict
        del data_dict
    
        ########### Fitting Data ############
        
        xdata = all_data_dict[str(ts)][xaxis]
        ydata = all_data_dict[str(ts)][yaxis]
        xdata_sorted = np.sort(xdata)
        ydata_sorted = ydata[np.argsort(xdata)]
        
        popt, pcov = curve_fit(Lorentzian, xdata_sorted, ydata_sorted, p0 = ([0.005, 1.25e-4, 1.64, 45]), bounds=([0.001, 1.e-5, 1.62, 10.], [0.1, 0.01, 1.66, 70.]))
        perr = np.sqrt(np.diag(pcov))
    
        # To compare with fits obtained during experiment, calculate another fit
        # characteristic, i.e. amplitude:
        amplitude = -popt[1]/ (popt[0]/2)**2
        # Now calculate percent differences with fits from lab day: Lab record
        # contains 'center', 'amplitude', 'offset', and 'FWHM' in that order.
        fp_dif = {}
        fp_dif['center'] = 100 * (fp[0] - popt[2]) / fp[0]
        fp_dif['amplitude'] = 100 * (fp[1] - amplitude) / fp[1]
        fp_dif['offset'] = 100 * (fp[2] - popt[3]) / fp[2]
        fp_dif['fwhm'] = 100 * (fp[3] - popt[0]) / fp[3]
    
        fit_chars = {}
        for descrip, values in zip(['parameters', 'errors', 'diffs'], [popt, perr, fp_dif]):
            fit_chars[descrip] = values
        all_fit_chars[str(ts)] = fit_chars
        
        ############ Plotting Data ##############
        if Plot_Data:
            fit_y = Lorentzian(xdata_sorted, popt[0], popt[1], popt[2], popt[3])
            
            fig = plt.figure()
            fig_plot = fig.add_subplot(111, xlabel=xaxis, ylabel=yaxis)
            fig_plot.plot(xdata, ydata, 'ko', xdata_sorted, fit_y, 'r-')
            title = 'electrode(s): ' + str(el) + ', offset(s): ' + str(of) + 'V, expected: ' + str(ef) + 'kHz, \n measured: ' + str(mf) + 'kHz, accuracy: ' + ac + ', note: ' + nt
            titles = fig_plot.set_title("\n".join(wrap(title + '\n' + fit_params)))
            fig.tight_layout()
            titles.set_y(1.05)
            fig.subplots_adjust(top=0.8)
            plot_name = str(ts) + '_' + yaxis + '_vs_' + xaxis
            fig.savefig(path_to_plot_data + plot_name + ext)
            plt.close(fig)
            del titles, fig_plot, fig, plot_name

        # An example plot for my report (Brennan)
        if ts == 190115:
            xmin = np.max(xdata)
            xmax = np.min(xdata)
            fit_x = np.linspace(xmin, xmax, 200)
            fit_y = Lorentzian(fit_x, popt[0], popt[1], popt[2], popt[3])

            plt.plot(xdata, ydata, 'ko', fit_x, fit_y, 'r', linewidth=2)
            plt.title('Example Data: Electrode 9, Offset -1 V', fontname='serif')
            plt.legend(['Data', 'Lorentzian fit'], loc=3)
            plt.xlabel('Tickle frequency [MHz]', fontname='serif')
            plt.ylabel('$^{40}$Ca$^+$ signal', fontname='serif')
            fig = plt.gcf()
            # This should get the legend as well:
            for o in fig.findobj(mpl.text.Text):
                o.set_fontname('serif')
            plt.savefig(path_to_plot_asys + 'example_data' + ext)
            plt.close()
            del xmin, xmax, fit_x, fit_y
        
        ########## Some terminal output ###########
        if Shell_Out:
            print('Timestamp:', str(ts) + ',', title)
            print('Computed fit parameters fwhm, scale, x0, y0, respectively:', all_fit_chars[str(ts)]['parameters'])
            print('Percent difference of computed fit parameters from those found on lab day: \n', all_fit_chars[str(ts)]['diffs'])
            print('Fit uncertainty in x0: ', all_fit_chars[str(ts)]['errors'][2])
    
        ##########################################
    
    
    ############ Analysis ##############
    # Write data from fits we care about to some new variables:
    f_meas = [0 for i in range(len(timestamps)-1)]
    fm_uncert = [0 for i in range(len(timestamps)-1)]
    for el, of, ts, i in zip(electrodes, offsets, timestamps, range(len(timestamps))):
        # Seems reasonable to put the zero offset value first:
        if of == 0.0:
            key = 'off. ' + str(of)
        else:
            key = 'elec. ' + str(el) + ', ' + 'off. ' + str(of)
        if i > 0 and i < 3:
            f_meas[i] = {key: all_fit_chars[str(ts)]['parameters'][2]}
            fm_uncert[i] = {key: all_fit_chars[str(ts)]['errors'][2]}
        elif i == 3:
            f_meas[0] = {key: all_fit_chars[str(ts)]['parameters'][2]}
            fm_uncert[0] = {key: all_fit_chars[str(ts)]['errors'][2]}
        elif i > 3:
            f_meas[i-1] = {key: all_fit_chars[str(ts)]['parameters'][2]}
            fm_uncert[i-1] = {key: all_fit_chars[str(ts)]['errors'][2]}
    
    # Calculate measured frequency shifts (I used the second measurement of the
    # frequency without any shifts instead of the first since the second one was a
    # more accurate measurement, but probably doesn't matter that much -- they're
    # both pretty close):
    delta_f_meas = []
    dfm_uncert = []
    labels_single = {}
    labels_single['electrodes'] = []
    labels_single['offsets'] = []
    labels_multi = {}
    labels_multi['electrodes'] = []
    labels_multi['offsets'] = []
    index_single = []
    index_multi = []
    index = 0
    for el, of, ts in zip(electrodes, offsets, timestamps):
        if of != 0.:
            if type(el) is int:
                labels_single['electrodes'].append(el)
                labels_single['offsets'].append(of)
                index_single.append(index)
                index += 1
            else:
                labels_multi['electrodes'].append(el)
                labels_multi['offsets'].append(of)
                index_multi.append(index)
                index += 1
            key = 'electrode(s): ' + str(el) + ', ' + 'offset: ' + str(of)
            delta_f_meas.append({key: all_fit_chars[str(ts)]['parameters'][2] - all_fit_chars['184523']['parameters'][2]})
            dfm_uncert.append({key: all_fit_chars[str(ts)]['errors'][2] - all_fit_chars['184523']['errors'][2]})

    # Run static potential offset test (make sure writing=True in 'analyze_waveform'):
    
    # Default inputs to 'static_potential_offset_test.py':
    electrode_offsets_ref = [[-0.15], [0.15], [-0.4], [0.4], [-1], [1], [-0.4], [0.4], [-1], [1], [-4], [4], [-4], [4], [-8.5], [8.5], [-8.5], [8.5], [-0.4], [0.4], [-0.4], [0.4], [-1], [1], [-1], [1], [-4], [4], [-4], [4], [-8.5], [8.5], [-8.5], [8.5], [-8.5,-8.5,-8.5], [8.5,8.5,8.5], [-8.5,-8.5,-8.5], [8.5,8.5,8.5], [-8.5,-8.5,-8.5], [8.5,8.5,8.5], [-8.5,-8.5,-8.5], [8.5,8.5,8.5]]
    
    electrode_combos_ref = [[7],[7],[6],[6],[5],[5],[8],[8],[9],[9],[4],[4],[10],[10],[3],[3],[11],[11],[21],[21],[23],[23],[20],[20],[24],[24],[19],[19],[25],[25],[18],[18],[26],[26],[0,1,2],[0,1,2],[12,13,14],[12,13,14],[15,16,17],[15,16,17],[27,28,29],[27,28,29]]

    electrode_offsets = electrode_offsets_ref
    electrode_combos = electrode_combos_ref
    if New_DEATH_Gain_Error:
        for ec_l, i in zip(electrode_combos, range(len(electrode_combos))):
            for ec, j in zip(ec_l, range(len(ec_l))):
                if physical_electrode_transform[ec] < 16: # These are the New DEATH channels
                    electrode_offsets[i][j] *= 1.027 # ~9.96V/9.70V
    if Fault_Analysis: # Account for the effects of electrode 10 not working
        for ec, i in zip(electrode_combos, range(len(electrode_offsets))):
            if ec[0] == 10:
                electrode_offsets[i][0] = Faulty_Voltage
    if Gain_Error_Analysis:
        for i in range(len(electrode_offsets)):
            for j in range(len(electrode_offsets[i])):
                electrode_offsets[i][j] *= Gain_Error

    with open('analysis_options.txt', 'w') as anop:
        if Gain_Error_Analysis: # Multiply all voltages by a constant
            anop.write('1\n') # True
        else:
            anop.write('0\n') # False
        anop.write(str(Gain_Error) + '\n')
        if Fault_Analysis: # Electrode 10 always produces a voltage value Faulty_Voltage
            anop.write('1\n') # True
        else:
            anop.write('0\n') # False
        anop.write(str(Faulty_Voltage) + '\n')
        if New_DEATH_Gain_Error:
            anop.write('1') # True
        else:
            anop.write('0') # False

    with open('spot_offset_inputs.csv', 'w') as spot_off_in:
        for eo in electrode_offsets:
            output = ''
            for of in eo:
                output += str(of) + ' '
            spot_off_in.write(output + '\n')

    with open('spot_combo_inputs.csv', 'w') as spot_com_in:
        for ec in electrode_combos:
            output = ''
            for co in ec:
                output += str(co) + ' '
            spot_com_in.write(output + '\n')
    
    os.system(py_command + ' static_potential_offset_test.py')
    delta_f_predicted = []
    f_predicted = []
    with open('delta_f_p.csv', 'r', newline='') as dfp:
        for row in csv.reader(dfp):
            delta_f_predicted.append(float(row[0])/1.e3) # kHz
    
    with open('f_p.csv', 'r', newline='') as fp:
        for row in csv.reader(fp):
            f_predicted.append(float(row[0])/1.e6) # MHz
    
    # Plot results compared with predictions for frequencies:
    xaxis_labels = tuple( [j for j in f_meas[i].keys()][0] for i in range(len(f_meas)) )
    fm = tuple( [j for j in f_meas[i].values()][0] for i in range(len(f_meas)) )
    fm_err = tuple( [j for j in fm_uncert[i].values()][0] for i in range(len(fm_uncert)) )
    
    write_data = False
    data_id = 'no_gain'
    # Run once with write_data = True:
    if write_data:
        os.system('touch ' + 'f_data_' + data_id + '.csv')
        with open('f_data_' + data_id + '.csv', 'w') as csvfile:
            datawriter = csv.writer(csvfile)
            for i in range(len(fm)):
                datawriter.writerow([xaxis_labels[i], str(fm[i]), str(fm_err[i]), str(f_predicted[i])])

    # After running once with write_data = True, run with write_data = False
    # to make a plot with the data you wrote, and some new data:
    if not write_data:

        with open('f_data_' + data_id + '.csv', 'r') as csvfile:
            datareader = csv.reader(csvfile)
            xaxis_labels_old, fm_old, fm_err_old, f_predicted_old = [], [], [], []
            for row in datareader:
                xaxis_labels_old.append(row[0])
                fm_old.append(float(row[1]))
                fm_err_old.append(float(row[2]))
                f_predicted_old.append(float(row[3]))

        fig = plt.figure(figsize=(8,8.5))
        topmar = 0.04
        botmar = 0.27
        rightmar = 0.02
        leftmar = 0.12
        gap = 0.02
        xlen = 1 - rightmar - leftmar
        ylen = 1 - botmar - gap - topmar

        ax1 = fig.add_axes([leftmar, botmar+gap+ylen/2.0, xlen, ylen/2.0])
        ax1.plot(range(1, len(xaxis_labels_old)+1), f_predicted_old, 'kd')
        ax1.errorbar(range(1, len(xaxis_labels_old)+1), fm_old, yerr=fm_err_old, fmt='co')
        ax1.xaxis.set_ticks(range(1, len(xaxis_labels_old)+1))
        ax1.xaxis.set_ticklabels(tuple('' for i in range(len(xaxis_labels_old)) ) )
        plt.grid()
        plt.title('Absolute Frequencies')

        ax2 = fig.add_axes([leftmar, botmar, xlen, ylen/2.0])
        ax2.plot(range(1, len(xaxis_labels)+1), f_predicted, 'kd')
        ax2.errorbar(range(1, len(xaxis_labels)+1), fm, yerr=fm_err, fmt='co')
        ax2.xaxis.set_ticks(range(1, len(xaxis_labels)+1))
        ax2.xaxis.set_ticklabels(xaxis_labels, size='small', rotation=90)
        plt.grid()

        fig.legend([ax2.get_children()[1], ax2.get_children()[4]], ['Predicted', 'Measured'], loc=3)
        plt.figtext(0.02, 0.68, 'Frequency [MHz]', fontsize=13, rotation='vertical')
        plt.figtext(0.93, 0.93, '(a)', fontsize=18)
        plt.figtext(0.93, 0.57, '(b)', fontsize=18)
        for o in fig.findobj(mpl.text.Text):
            o.set_fontname('serif')

        fig.savefig(path_to_plot_asys + 'f_plot' + ext)
        plt.close(fig)
        del topmar, botmar, rightmar, leftmar, gap, xlen, ylen

        os.system('cd plots/analysis/ && evince f_plot.pdf')
    
    # Plot results compared with predictions for delta_freqs:
    xaxis_labels = tuple( [j for j in delta_f_meas[i].keys()][0] for i in range(len(delta_f_meas)) )
    dfm = np.array( [[j for j in delta_f_meas[i].values()][0]*1.e3 for i in range(len(delta_f_meas))] )
    dfm_err = np.array( [[j for j in dfm_uncert[i].values()][0]*1.e3 for i in range(len(dfm_uncert))] )
    
    plt.plot(range(1, len(xaxis_labels)+1), delta_f_predicted, 'kd')
    plt.errorbar(range(1, len(xaxis_labels)+1), dfm, yerr=dfm_err, fmt='co')
    plt.xticks(range(1, len(xaxis_labels)+1), xaxis_labels, size='small', rotation=90)
    plt.tight_layout()
    plt.savefig(path_to_plot_asys + 'delta_f_plot' + ext)
    plt.close()

    ########## Making nicer plot for report starts here ##########

    # Sort labels for single electrode shifts:
    electrode_labels = np.sort( np.array(labels_single['electrodes']) )
    electrode_labels += 1 # make electrodes 1-indexed instead of 0-indexed
    offset_labels = np.array(labels_single['offsets'])[np.argsort( np.array(labels_single['electrodes']) )]

    # Keep track of how the labels are sorted, so we can sort the data the same way after:
    index_track = np.array(index_single)[np.argsort( np.array(labels_single['electrodes']) )]

    # Start splitting up the LABELS for the different subplots:
    electrode_labels_p, electrode_labels_n, offset_labels_p, offset_labels_n = [], [], [], []
    index_track_p, index_track_n = [], []
    # First split negative and positive offset cases:
    for el, ol, it in zip(electrode_labels, offset_labels, index_track):
        if ol > 0.0:
            electrode_labels_p.append(el)
            offset_labels_p.append(ol)
            index_track_p.append(it)
        elif ol < 0.0:
            electrode_labels_n.append(el)
            offset_labels_n.append(ol)
            index_track_n.append(it)
    # Next split top and bottom electrodes:
    electrode_labels_top_p, electrode_labels_bot_p, offset_labels_top_p, offset_labels_bot_p = [], [], [], []
    index_track_top_p, index_track_bot_p = [], []
    for elp, olp, itp in zip(electrode_labels_p, offset_labels_p, index_track_p):
        if elp < 16:
            electrode_labels_top_p.append(elp)
            offset_labels_top_p.append(olp)
            index_track_top_p.append(itp)
        elif elp > 15:
            electrode_labels_bot_p.append(elp)
            offset_labels_bot_p.append(olp)
            index_track_bot_p.append(itp)

    electrode_labels_top_n, electrode_labels_bot_n, offset_labels_top_n, offset_labels_bot_n = [], [], [], []
    index_track_top_n, index_track_bot_n = [], []
    for eln, oln, itn in zip(electrode_labels_n, offset_labels_n, index_track_n):
        if eln < 16:
            electrode_labels_top_n.append(eln)
            offset_labels_top_n.append(oln)
            index_track_top_n.append(itn)
        elif eln > 15:
            electrode_labels_bot_n.append(eln)
            offset_labels_bot_n.append(oln)
            index_track_bot_n.append(itn)

    # Now make multi electrode cases 1-indexed instead of 0-indexed:
    for i in range(len(labels_multi['electrodes'])):
        for j in range(len(labels_multi['electrodes'][i])):
            labels_multi['electrodes'][i][j] += 1

    # And add multi electrode cases in manually:
    electrode_labels_top_n.insert(0, labels_multi['electrodes'][0])
    electrode_labels_top_p.insert(0, labels_multi['electrodes'][1])
    electrode_labels_top_n.append(labels_multi['electrodes'][2])
    electrode_labels_top_p.append(labels_multi['electrodes'][3])
    electrode_labels_bot_n.insert(0, labels_multi['electrodes'][4])
    electrode_labels_bot_p.insert(0, labels_multi['electrodes'][5])
    electrode_labels_bot_n.append(labels_multi['electrodes'][6])
    electrode_labels_bot_p.append(labels_multi['electrodes'][7])

    offset_labels_top_n.insert(0, labels_multi['offsets'][0])
    offset_labels_top_p.insert(0, labels_multi['offsets'][1])
    offset_labels_top_n.append(labels_multi['offsets'][2])
    offset_labels_top_p.append(labels_multi['offsets'][3])
    offset_labels_bot_n.insert(0, labels_multi['offsets'][4])
    offset_labels_bot_p.insert(0, labels_multi['offsets'][5])
    offset_labels_bot_n.append(labels_multi['offsets'][6])
    offset_labels_bot_p.append(labels_multi['offsets'][7])

    index_track_top_n.insert(0, index_multi[0])
    index_track_top_p.insert(0, index_multi[1])
    index_track_top_n.append(index_multi[2])
    index_track_top_p.append(index_multi[3])
    index_track_bot_n.insert(0, index_multi[4])
    index_track_bot_p.insert(0, index_multi[5])
    index_track_bot_n.append(index_multi[6])
    index_track_bot_p.append(index_multi[7])

#    for oltn, i in zip(offset_labels_top_n, range( len(offset_labels_top_n) )):
#        offset_labels_top_n[i] = round(oltn, 2)
#    for oltp, i in zip(offset_labels_top_p, range( len(offset_labels_top_p) )):
#        offset_labels_top_p[i] = round(oltp, 2)
#    for olbn, i in zip(offset_labels_bot_n, range( len(offset_labels_bot_n) )):
#        offset_labels_bot_n[i] = round(olbn, 2)
#    for olbp, i in zip(offset_labels_bot_p, range( len(offset_labels_bot_p) )):
#        offset_labels_bot_p[i] = round(olbp, 2)

    delta_f_fig = plt.figure(figsize=(7,9))
    botmargin = 0.12
    topmargin = 0.1
    leftmargin = 0.1
    rightmargin = 0.05
    gap = 0.07
    ylen = 1 - botmargin - topmargin - 3*gap
    xlen = 1 - leftmargin - rightmargin
    ofcolor = (0,0,0.6)

    ax1 = delta_f_fig.add_axes([leftmargin, botmargin+2*ylen/3+3*gap, xlen, ylen/3])
    ax1.bar(np.arange(1.0, 2.2*len(electrode_labels_top_p)+1.0, 2.2), np.array(delta_f_predicted)[index_track_top_p], color='k', width=0.8)
    ax1.bar(np.arange(1.8, 2.2*len(electrode_labels_top_p)+1.8, 2.2), dfm[index_track_top_p], yerr=dfm_err[index_track_top_p], color='c', ecolor='k', width=0.8)

    ax1.xaxis.set_ticks(np.arange(1.8, 2.2*len(electrode_labels_top_p)+1.8, 2.2))
    ax1.xaxis.set_ticklabels(electrode_labels_top_p)
    ax1top = ax1.twiny()
    ax1top.xaxis.set_ticks(np.arange(1.8, 2.2*len(electrode_labels_top_p)+1.8, 2.2))
    ax1top.set_xlim(ax1.get_xlim())
    ax1top.xaxis.set_ticklabels(offset_labels_top_p)
    ax1.xaxis.set_label_text('Electrodes')
    ax1top.xaxis.set_label_text('Offsets [V]')
    label1top = ax1top.xaxis.get_label()
    label1top.set_color(ofcolor)
    for tick1, tick2 in zip(ax1.xaxis.get_major_ticks(), ax1top.xaxis.get_major_ticks()):
        tick1.label1.set_fontsize(10)
        tick2.label2.set_fontsize(10)
        tick2.label2.set_color(ofcolor)


    ax2 = delta_f_fig.add_axes([leftmargin, botmargin+ylen/3+2*gap, xlen, ylen/3])
    ax2.bar(np.arange(1.0, 2.2*len(electrode_labels_top_n)+1.0, 2.2), np.array(delta_f_predicted)[index_track_top_n], color='k', width=0.8)
    ax2.bar(np.arange(1.8, 2.2*len(electrode_labels_top_n)+1.8, 2.2), dfm[index_track_top_n], yerr=dfm_err[index_track_top_n], color='c', ecolor='k', width=0.8)
    ax2.xaxis.set_ticks(np.arange(1.8, 2.2*len(electrode_labels_top_n)+1.8, 2.2))
    ax2.xaxis.set_ticklabels(offset_labels_top_n)
    ax2top = ax2.twiny()
    ax2.xaxis.set_label_text('Offsets [V]')
    label2 = ax2.xaxis.get_label()
    label2.set_color(ofcolor)
    ax2top.xaxis.set_ticks(np.arange(1.8, 2.2*len(electrode_labels_top_n)+1.8, 2.2))
    ax2top.xaxis.set_ticklabels(electrode_labels_top_n)
    ax2top.set_xlim(ax2.get_xlim())
    for tick1, tick2 in zip(ax2.xaxis.get_major_ticks(), ax2top.xaxis.get_major_ticks()):
        tick1.label1.set_fontsize(10)
        tick1.label1.set_color(ofcolor)
        tick2.label2.set_fontsize(10)


    ax3 = delta_f_fig.add_axes([leftmargin, botmargin+ylen/6+gap, xlen, ylen/6])
    ax3.bar(np.arange(1.0, 2.2*len(electrode_labels_bot_p)+1.0, 2.2), np.array(delta_f_predicted)[index_track_bot_p], color='k', width=0.8)
    ax3.bar(np.arange(1.8, 2.2*len(electrode_labels_bot_p)+1.8, 2.2), dfm[index_track_bot_p], yerr=dfm_err[index_track_bot_p], color='c', ecolor='k', width=0.8)
    ax3.xaxis.set_ticks(np.arange(1.8, 2.2*len(electrode_labels_bot_p)+1.8, 2.2))
    ax3.xaxis.set_ticklabels(electrode_labels_bot_p)
    ax3top = ax3.twiny()
    ax3top.xaxis.set_ticks(np.arange(1.8, 2.2*len(electrode_labels_bot_p)+1.8, 2.2))
    ax3top.set_xlim(ax3.get_xlim())
    ax3top.xaxis.set_ticklabels(offset_labels_bot_p)
    ax3.xaxis.set_label_text('Electrodes')
    for tick1, tick2 in zip(ax3.xaxis.get_major_ticks(), ax3top.xaxis.get_major_ticks()):
        tick1.label1.set_fontsize(10)
        tick2.label2.set_fontsize(10)
        tick2.label2.set_color(ofcolor)
    ax3.yaxis.set_ticks([0., 5., 10., 15.])


    ax4 = delta_f_fig.add_axes([leftmargin, botmargin, xlen, ylen/6])
    ax4.bar(np.arange(1.0, 2.2*len(electrode_labels_bot_n)+1.0, 2.2), np.array(delta_f_predicted)[index_track_bot_n], color='k', width=0.8)
    ax4.bar(np.arange(1.8, 2.2*len(electrode_labels_bot_n)+1.8, 2.2), dfm[index_track_bot_n], yerr=dfm_err[index_track_bot_n], color='c', ecolor='k', width=0.8)
    ax4.xaxis.set_ticks(np.arange(1.8, 2.2*len(electrode_labels_bot_n)+1.8, 2.2))
    ax4.xaxis.set_ticklabels(offset_labels_bot_n)
    ax4top = ax4.twiny()
    ax4.xaxis.set_label_text('Offsets [V]')
    label4 = ax4.xaxis.get_label()
    label4.set_color(ofcolor)
    ax4top.xaxis.set_ticks(np.arange(1.8, 2.2*len(electrode_labels_bot_n)+1.8, 2.2))
    ax4top.set_xlim(ax4.get_xlim())
    ax4top.xaxis.set_ticklabels(electrode_labels_bot_n)
    for tick1, tick2 in zip(ax4.xaxis.get_major_ticks(), ax4top.xaxis.get_major_ticks()):
        tick1.label1.set_fontsize(10)
        tick1.label1.set_color(ofcolor)
        tick2.label2.set_fontsize(10)
    ax4.yaxis.set_ticks([-15., -10., -5., 0.])

    yt1, yt2, yt3, yt4 = ax1.yaxis.get_major_ticks(), ax2.yaxis.get_major_ticks(), ax3.yaxis.get_major_ticks(), ax4.yaxis.get_major_ticks()
    for yt in [yt1, yt2, yt3, yt4]:
        for ytck in yt:
            ytck.label1.set_fontsize(10)


    delta_f_fig.text(0.02, 0.58, 'Frequency shifts [kHz]', fontsize=12, rotation='vertical')
    delta_f_fig.legend((ax1.get_children()[1], ax1.get_children()[14]), ('Predicted', 'Measured'), (0.1, 0.04), fontsize=11)
    delta_f_fig.suptitle('Relative Frequency Shifts', fontsize=14)

    # Set the font for the whole figure:
    parents = delta_f_fig.get_children()
    while len(parents) > 0:
        new_parents = []
        for parent in parents:
            if type(parent) == mpl.text.Text:
                parent.set_fontname('serif')
            else:
                children = []
                try:
                    children = parent.get_children()
                except AttributeError:
                    continue
                new_parents.extend(children)
        parents = new_parents


    delta_f_fig.savefig(path_to_plot_asys + 'delta_f_plot_nice' + ext)
    # plt.show()
    plt.close(delta_f_fig)
    
    if Open_Plots: # This only opens analysis plots not data plots
        # os.system('cd ' + path_to_plot_asys + ' && ' + plot_viewer + ' f_plot' + ext)
        os.system('cd ' + path_to_plot_asys + ' && ' + plot_viewer + ' delta_f_plot_nice' + ext)
    

if __name__ == "__main__":
    Analyze(Plot_Data=False, Shell_Out=False, Open_Plots=False, New_DEATH_Gain_Error=True, Gain_Error_Analysis=True, Gain_Error=1.047, Fault_Analysis=True, Faulty_Voltage=0.0)
