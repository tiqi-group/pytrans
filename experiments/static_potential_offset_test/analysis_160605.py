# Data analysis script for static potential offset experiment on June 5th 2016.

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
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

ext = '.png' # whatever file type you want to save your plot figures as
plot_viewer = 'viewnior'
py_command = 'python3.4' # the command to run python in shell for your system

# what you want to plot on the x and y axes -- needs to be a string contained
# in the fist row of the data file (the first row of the data file contains
# the descriptions of the data):
xaxis = 'x' # ' y' ' time [s]'
yaxis = ' Ca+ signal (s)' # 'DDS err.' ' BkgCorr counts' ' Timing err.'


# Contents of the table we filled in on the day of the experiments:
timestamps = (183353, 183622, 183731, 184523, 184806, 185005, 185412, 185602, 185807, 185953, 190115, 190241, 190423, 190602, 190831, 191224, 191501, 191748, 191925, 192056, 192307, 192442, 193508, 193906, 194209, 194402, 194601, 194723, 194956, 195114, 195330, 195633, 195908, 200058, 200303, 200709, 202753, 202954, 203127, 203245, 203411, 203531, 203655, 203822)
electrodes = (7,7,7,6,6,6,5,5,8,8,9,9,4,4,10,10,3,3,11,11,21,21,23,23,20,20,24,24,19,19,25,25,18,18,26,26,(0,1,2),(0,1,2),(12,13,14),(12,13,14),(15,16,17),(15,16,17),(27,28,29),(27,28,29))
offsets = (0., -0.15, 0.15, 0., -0.4, 0.4, -1, 1, -0.4, 0.4, -1, 1, -4, 4, -4, 4, -8.5, 8.5, -8.5, 8.5, -0.4, 0.4, -0.4, 0.4, -1, 1, -1, 1, -4, 4, -4, 4, -8.5, 8.5, -8.5, 8.5, -8.5, 8.5, -8.5, 8.5, -8.5, 8.5, -8.5, 8.5)
expected_freqs = (1598, 1610, 1586, 1598, 1589, 1606, 1589, 1607, 1589, 1607, 1589, 1607, 1589, 1607, 1589, 1607, 1592, 1604, 1592, 1604, 1586, 1610, 1586, 1610, 1589, 1607, 1589, 1607, 1589.5, 1606.5, 1590, 1606, 1592, 1604, 1592, 1604, -1, -1, -1, -1, -1, -1, -1, -1) # kHz
meas_freqs = (1641, 1655, 1629, 1642, 1627, 1656, 1634, 1649, 1630, 1652, 1632, 1650, 1635, 1649, 1642, 1642, 1638, 1646, 1637, 1647, 1630, 1653, 1627, 1656, 1633, 1651, 1634, 1651, 1634, 1651, 1635, 1650, 1638, 1648, 1637, 1648, 1642, 1643, 1641, 1643, 1642, 1643, 1642, 1643) # kHz
accurs = ('', '', '', '', 'good', 'good', 'good', 'decent', 'good', 'decent', 'decent', 'decent', 'decent', 'decent', 'decent', 'decent', 'decent', 'sosolala', 'good', 'sosolala', 'decent', 'sosolala', 'decent', 'decent', 'decent', 'sosolala', 'decent', 'decent', 'decent', 'sosolala', 'good', 'sosolala', 'decent', 'sosolala', 'decent', 'sosolala', 'decent', 'decent', 'decent', 'decent', 'decent', 'good', 'decent', 'decent')
notes = ('', '', '', '', '', '', '', '', '', '', '', '', '', '', 'NO EFFECT', 'NO EFFECT', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '')

assert len(timestamps)==len(electrodes)==len(offsets)==len(expected_freqs)==len(meas_freqs)==len(accurs)==len(notes), 'check yourself before you wreck yourself'

# To use for fitting data below:
def Lorentzian(x, fwhm, scale, x0, y0):
        return y0 - (scale / ((x - x0)**2 + (fwhm/2)**2))

def Analyze(Plot_Data=True, Shell_Out=True, Open_Plots=True, Faulty_Voltage=0.0):
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
        
        ############ Plotting ##############
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
        
        ###### Some info for the terminal #######
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
        if i > 0 and i < 3:
            key = 'electrode(s): ' + str(el) + ', ' + 'offset: ' + str(of)
            f_meas[i] = {key: all_fit_chars[str(ts)]['parameters'][2]}
            fm_uncert[i] = {key: all_fit_chars[str(ts)]['errors'][2]}
        elif i == 3:
            key = 'electrode(s): ' + str(el) + ', ' + 'offset: ' + str(of)
            f_meas[0] = {key: all_fit_chars[str(ts)]['parameters'][2]}
            fm_uncert[0] = {key: all_fit_chars[str(ts)]['errors'][2]}
        elif i > 3:
            key = 'electrode(s): ' + str(el) + ', ' + 'offset: ' + str(of)
            f_meas[i-1] = {key: all_fit_chars[str(ts)]['parameters'][2]}
            fm_uncert[i-1] = {key: all_fit_chars[str(ts)]['errors'][2]}
    
    # Calculate measured frequency shifts (I used the second measurement of the
    # frequency without any shifts instead of the first since the second one was a
    # more accurate measurement, but probably doesn't matter that much -- they're
    # both pretty close):
    delta_f_meas = []
    dfm_uncert = []
    for el, of, ts in zip(electrodes, offsets, timestamps):
        if of != 0.:
            key = 'electrode(s): ' + str(el) + ', ' + 'offset: ' + str(of)
            delta_f_meas.append({key: all_fit_chars[str(ts)]['parameters'][2] - all_fit_chars['184523']['parameters'][2]})
            dfm_uncert.append({key: all_fit_chars[str(ts)]['errors'][2] - all_fit_chars['184523']['errors'][2]})
    
    # Run static potential offset test (make sure writing=True in 'analyze_waveform'):
    
    # Default inputs to 'static_potential_offset_test.py':
    electrode_offsets_ref = [[-0.15], [0.15], [-0.4], [0.4], [-1], [1], [-0.4], [0.4], [-1], [1], [-4], [4], [-4], [4], [-8.5], [8.5], [-8.5], [8.5], [-0.4], [0.4], [-0.4], [0.4], [-1], [1], [-1], [1], [-4], [4], [-4], [4], [-8.5], [8.5], [-8.5], [8.5], [-8.5,-8.5,-8.5], [8.5,8.5,8.5], [-8.5,-8.5,-8.5], [8.5,8.5,8.5], [-8.5,-8.5,-8.5], [8.5,8.5,8.5], [-8.5,-8.5,-8.5], [8.5,8.5,8.5]]
    
    electrode_combos_ref = [[7],[7],[6],[6],[5],[5],[8],[8],[9],[9],[4],[4],[10],[10],[3],[3],[11],[11],[21],[21],[23],[23],[20],[20],[24],[24],[19],[19],[25],[25],[18],[18],[26],[26],[0,1,2],[0,1,2],[12,13,14],[12,13,14],[15,16,17],[15,16,17],[27,28,29],[27,28,29]]

    electrode_offsets = []
    electrode_combos = []
    for eor, ecr in zip(electrode_offsets_ref, electrode_combos_ref):
        if ecr[0] == 10:
            eor[0] = Faulty_Voltage
        else:
            eor.append(Faulty_Voltage)
            ecr.append(10)
        electrode_offsets.append(eor)
        electrode_combos.append(ecr)

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
            delta_f_predicted.append(float(row[0])/1.e3)
    
    with open('f_p.csv', 'r', newline='') as fp:
        for row in csv.reader(fp):
            f_predicted.append(float(row[0])/1.e6)
    
    # Plot results compared with predictions for frequencies:
    xaxis_labels = tuple( [j for j in f_meas[i].keys()][0] for i in range(len(f_meas)) )
    fm = np.array([[j for j in f_meas[i].values()][0] for i in range(len(f_meas))] )
    fm_err = np.array([[j for j in fm_uncert[i].values()][0] for i in range(len(fm_uncert))] )
    
    plt.plot(range(len(xaxis_labels)), np.array(f_predicted), 'rd')
    plt.errorbar(range(len(xaxis_labels)), fm, yerr=fm_err, fmt='ko')
    plt.xticks(range(len(xaxis_labels)), xaxis_labels, size='small', rotation=90)
    plt.tight_layout()
    plt.savefig(path_to_plot_asys + 'f_plot' + ext)
    plt.close()
    
    # Plot results compared with predictions for delta_freqs:
    xaxis_labels = tuple( [j for j in delta_f_meas[i].keys()][0] for i in range(len(delta_f_meas)) )
    dfm = np.array([[j for j in delta_f_meas[i].values()][0]*1.e3 for i in range(len(delta_f_meas))] )
    dfm_err = np.array([[j for j in dfm_uncert[i].values()][0]*1.e3 for i in range(len(dfm_uncert))] )
    
    plt.plot(range(len(xaxis_labels)), np.array(delta_f_predicted), 'rd')
    plt.errorbar(range(len(xaxis_labels)), dfm, yerr=dfm_err, fmt='ko')
    plt.xticks(range(len(xaxis_labels)), xaxis_labels, size='small', rotation=90)
    plt.tight_layout()
    plt.savefig(path_to_plot_asys + 'delta_f_plot' + ext)
    plt.close()
    
    if Open_Plots: # This only opens analysis plots not data plots!
        os.system('cd ' + path_to_plot_asys + ' && ' + plot_viewer + ' f_plot' + ext)
        os.system('cd ' + path_to_plot_asys + ' && ' + plot_viewer + ' delta_f_plot' + ext)
    

if __name__ == "__main__":
    Analyze(Plot_Data=False, Shell_Out=False, Open_Plots=True, Faulty_Voltage=0.0)
