#!/usr/bin/env python3
#
# Simple least-squares fitting script to take DEATH calibration values
# and calculate the required gains and offsets

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import pdb
st = pdb.set_trace

#if __name__ == "__main__":
#    data_file = 'example_death_cal.csv'
#    dd = pd.read_csv(data_file)

# Measured on DEATH channels, 1->16; 2016.03.18
# max_voltages = [-10, 10]
# exact_voltages = np.array([-8, -4, 0, 4, 8])
# cal_voltages = np.array([[-7.66, -3.91, -0.07, 3.71, 7.46],
#                          [-7.59, -3.87, -0.07, 3.67, 7.39],
#                          [-7.601, -3.89, -0.08, 3.69, 7.40],
#                          [-7.59, -3.89, -0.09, 3.67, 7.38],

#                          [-7.64, -3.91, -0.08, 3.69, 7.43],
#                          [-7.57, -3.87, -0.08, 3.65, 7.35],
#                          [-7.62, -3.91, -0.09, 3.69, 7.42],
#                          [-7.57, -3.88, -0.09, 3.66, 7.36],

#                              [-7.68, -3.93, -0.08, 3.71, 7.46],
#                          [-7.63, -3.91, -0.08, 3.69, 7.42],
#                          [-7.59, -3.89, -0.09, 3.67, 7.38],
#                          [-7.56, -3.87, -0.08, 3.66, 7.35],

#                              [-7.64, -3.91, -0.08, 3.69 ,7.42],
#                          [-7.62, -3.90, -0.08, 3.67, 7.40],
#                          [-7.66, -3.92, -0.07, 3.73, 7.47],
#                          [-7.61, -3.90, -0.09, 3.68, 7.40]])

# Measured on DEATH channels, 1->16; old DEATHs only -- 2016.04.14
max_voltages = [-10, 10]
exact_voltages = np.array([-8, -5, -2, 0, 2, 5, 8])
cal_voltages = np.array([[-7.37, -4.61, -1.87, -0.04, 1.76, 4.51, 7.26],
                         [-7.34, -4.60, -1.87, -0.05, 1.74, 4.47, 7.21],
                         [-7.38, -4.63, -1.88, -0.04, 1.78, 4.53, 7.29],
                         [-7.35, -4.62, -1.88, -0.06, 1.75, 4.49, 7.22],

                         [-7.43, -4.66, -1.91, -0.08, 1.73, 4.48, 7.25],
                         [-7.26, -4.53, -1.82, -0.01, 1.77, 4.48, 7.21],
                         [-7.37, -4.61, -1.86, -0.02, 1.81, 4.56, 7.32],
                         [-7.31, -4.59, -1.87, -0.05, 1.75, 4.47, 7.19],

                         [-7.43, -4.64, -1.88, -0.03, 1.79, 4.56, 7.34],
                         [-7.38, -4.63, -1.89, -0.06, 1.74, 4.48, 7.23],
                         [-7.38, -4.63, -1.88, -0.05, 1.77, 4.52, 7.27],
                         [-7.39, -4.65, -1.91, -0.09, 1.72, 4.46, 7.20],

                         [-7.39, -4.63, -1.88, -0.05, 1.76, 4.50, 7.27],
                         [-7.34, -4.59, -1.86, -0.04, 1.75, 4.49, 7.23],
                         [-7.41, -4.65, -1.88, -0.04, 1.79, 4.55, 7.31],
                         [-7.25, -4.55, -1.84, -0.04, 1.75, 4.46, 7.15]]);
                         
#cal_outputs = np.zeros([2, 16])
#st()
for k, cv in enumerate(cal_voltages):
    lin_result = linregress(exact_voltages, cv)
    slope_correction = 1.0/lin_result.slope
    offset_correction = max_voltages[0] + max_voltages[1]*lin_result.slope - lin_result.intercept
    print('cal for Ch ' + str(k+1), int(np.round(slope_correction*32768)), int(np.round(offset_correction*65536/20)))
    if not np.mod(k+1,4):
        print("\n")
        # slope_correction = 32768*(1.0/lin_result.slope)
        # offset_correction = (65536//20)*(-10*lin_result.slope+lin_result.intercept)
        # offset_correction = (65536//20)*(-lin_result.intercept/lin_result.slope)


# lin_offset_result = linregress(exact_voltages, cal_v_offset)


# st()
