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

cal_voltages = np.array([
    [-8.2399,-5.1600,-2.0967,-0.046,1.9893,5.0537,8.1343],
    [-8.2082,-5.1465,-2.1007,-0.06215,1.9612,5.0067,8.0681],
    [-8.2583,-5.1802,-2.1012,-0.049,2.0022,5.0815,8.1600],
    [-8.223,-5.165,-2.106,-0.067,1.971,5.030,8.089],
    
    [-8.303,-5.2146,-2.1406,-0.084,1.9581,5.032,8.121],
    [-8.1264,-5.0806,-2.0504,-0.0223,1.9905,5.0214,8.0681],
    [-8.2501,-5.1667,-2.0829,-0.027,2.0271,5.1111,8.1945],
    [-8.1712,-5.1304,-2.0887,-0.0616,1.9652,5.0071,8.0480],
    
    [-8.3085,-5.2008,-2.1088,-0.0395,2.0146,5.1068,8.2150],
    [-8.2508,-5.1762,-2.1160,-0.0680,1.9648,5.0253,8.1000],
    [-8.2589,-5.1852,-2.1101,-0.061,1.9881,5.0638,8.1374],
    [-8.2603,-5.2006,-2.1387,-0.0988,1.9419,5.0040,8.0638],
    
    [-8.2625,-5.1787,-2.1096,-0.056,1.9827,5.0518,8.1350],
    [-8.2065,-5.1415,-2.0907,-0.0495,1.9774,5.0281,8.0937],
    [-8.2917,-5.2008,-2.1100,-0.0495,2.0095,5.1003,8.1913],
    [-8.1152,-5.0909,-2.0644,-0.0483,1.9687,4.9954,8.0196],

    [-8.0567,-5.0565,-2.0711,-0.0717,1.9106,4.8962,7.8958],
    [-8.0306,-5.0413,-2.0662,-0.0735,1.9022,4.8774,7.8664],
    [-8.0819,-5.0769,-2.0705,-0.0664,1.9378,4.9441,7.9492],
    [-8.0277,-5.0470,-2.0653,-0.0779,1.9104,4.8920,7.8729],

    [-8.0892,-5.0754,-2.0762,-0.0690,1.9231,4.9223,7.9357],
    [-8.0446,-5.0480,-2.0655,-0.0692,1.9119,4.8942,7.8904],
    [-8.0003,-5.0297,-2.0559,-0.0729,1.9082,4.8810,7.8528],
    [-7.9620,-5.0040,-2.0446,-0.0718,1.9012,4.8606,7.8187],

    [-8.0473,-5.0478,-2.0630,-0.0641,1.9177,4.9023,7.9010],
    [-7.9605,-4.9940,-2.0413,-0.0677,1.8944,4.8473,7.8137],
    [-8.0334,-5.0485,-2.0625,-0.0725,1.9183,4.9040,7.8887],
    [-7.9722,-5.0107,-2.0477,-0.0716,1.9034,4.8664,7.8278],

    [-8.0638,-5.0576,-2.0648,0.0623,1.9256,4.9191,7.9262],
    [-7.9829,-5.0071,-2.0450,0.00628,1.9046,4.866,7.8424],
    [-8.0076, -5.0328, -2.0567, -0.0724, 1.9113, 4.8870, 7.8619],
    #[-8.0076,-5.0327,-2.0566,0.0724,1.9113,5.8872,7.8619],
    #[-7.9949,-5.0273,-2.0580,0.0785,1.9011,4.8698,7.8372]
    [-7.9951, -5.0270, -2.0579, -0.0785, 1.9011, 4.8699, 7.8370]
    ])

# Measured on DEATH channels, 1->16; old DEATHs only -- 2016.04.14
max_voltages = np.array([9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9,
                9.5,9.5,9.5,9.5, 9.5,9.5,9.5,9.5, 9.5,9.5,9.5,9.5, 9.5,9.5,9.5,9.5]);
min_voltages = -max_voltages
exact_voltages = np.array([-8, -5, -2, 0, 2, 5, 8])
                         
#cal_outputs = np.zeros([2, 16])
#st()
for k, (cv, mv) in enumerate(zip(cal_voltages, max_voltages)):
    lin_result = linregress(exact_voltages, cv)
    a, b = lin_result.slope, lin_result.intercept
    a2, b2 = np.polyfit(exact_voltages, cv, 1)
    G = mv * a / 32768
    mb = mv/G
    H = b/G - 32768
    ob = -H - mb
    # ob = -b/G + mb - mb
    # st()
    # slope_correction = 1.0/lin_result.slope
    # offset_correction = max_voltages[0] + max_voltages[1]*lin_result.slope - lin_result.intercept
    print('cal for Ch ' + str(k+1), int(np.round(mb)), int(np.round(ob)))
    #print('G, H for Ch ' + str(k+1), G, H)

    def vof(vif, Geff=G, mbeff=mb, vmax=mv, obeff=ob, Heff=H):
        return Geff*(mbeff*(vif/vmax + 1) + obeff + Heff)

    if False:
        plt.plot(exact_voltages, cv,'x')
        plt.plot([-mv, mv],[-a*mv+b,a*mv+b],'b')
        plt.grid(True)
        plt.show()
    
    if k == 0:
        pass
        
    if not np.mod(k+1,4):
        print("\n")
        # slope_correction = 32768*(1.0/lin_result.slope)
        # offset_correction = (65536//20)*(-10*lin_result.slope+lin_result.intercept)
        # offset_correction = (65536//20)*(-lin_result.intercept/lin_result.slope)


# lin_offset_result = linregress(exact_voltages, cal_v_offset)


# st()
