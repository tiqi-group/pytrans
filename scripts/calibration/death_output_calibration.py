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

if __name__ == "__main__":

    # 2016_04_22; both pre-cal and post-cal voltages
    cal_voltages = np.array([
        [-8.2418,-5.1623,-2.0981,-0.0473,1.9887,5.054,8.1349],
        [-8.2102,-5.148,-2.1015,-0.0627,1.9612,5.0077,8.07],
        [-8.2588,-5.181,-2.1019,-0.0501,2.0015,5.0812,8.1602],
        [-8.2225,-5.165,-2.1066,-0.0683,1.9701,5.0297,8.0874],

        [-8.3007,-5.2135,-2.1402,-0.0846,1.9573,5.0307,8.1188],
        [-8.1253,-5.0801,-2.0506,-0.0229,1.9898,5.0206,8.0667],
        [-8.2483,-5.166,-2.0833,-0.0282,2.0261,5.1099,8.1929],
        [-8.1719,-5.1313,-2.0895,-0.0624,1.9646,5.0068,8.0476],

        [-8.3099,-5.2018,-2.1095,-0.0401,2.0141,5.1069,8.2155],
        [-8.2535,-5.1782,-2.1171,-0.0691,1.9645,5.0261,8.1015],
        [-8.2616,-5.1875,-2.1115,-0.062,1.9877,5.0649,8.14],
        [-8.2634,-5.2027,-2.1398,-0.0994,1.9419,5.0052,8.0663],

        [-8.2625,-5.1787,-2.1093,-0.0561,1.9833,5.0534,8.1378],
        [-8.2096,-5.1431,-2.0912,-0.0491,1.9787,5.0312,8.0982],
        [-8.2959,-5.2033,-2.1108,-0.0493,2.0109,5.104,8.197],
        [-8.1203,-5.094,-2.0656,-0.0483,1.97,4.9992,8.0259],


        [-8.0563,-5.0562,-2.071,-0.0717,1.9105,4.8959,7.8955],
        [-8.0307,-5.0414,-2.0663,-0.0736,1.9022,4.8774,7.8663],
        [-8.0825,-5.0773,-2.0706,-0.0664,1.938,4.9446,7.9501],
        [-8.0286,-5.0477,-2.0656,-0.0781,1.9107,4.8929,7.8741],

        [-8.0888,-5.0752,-2.0762,-0.0691,1.923,4.9222,7.9356],
        [-8.0441,-5.0477,-2.0656,-0.0694,1.9117,4.894,7.8899],
        [-7.9997,-5.0285,-2.0559,-0.0732,1.908,4.8807,7.8525],
        [-7.9614,-5.0038,-2.0446,-0.072,1.9009,4.8602,7.8181],

        [-8.0466,-5.0474,-2.0629,-0.0642,1.9175,4.9019,7.9006],
        [-7.9595,-4.9934,-2.041,-0.0678,1.894,4.8466,7.8124],
        [-8.0316,-5.0475,-2.0621,-0.0726,1.9179,4.9031,7.8873],
        [-7.9708,-5.0099,-2.0474,-0.0717,1.9031,4.8655,7.8265],

        [-8.0635,-5.0571,-2.0645,-0.0622,1.9256,4.9186,7.9251],
        [-7.9822,-5.0066,-2.045,-0.0629,1.904,4.8659,7.8415],
        [-8.0067,-5.0322,-2.0566,-0.0726,1.9109,4.8862,7.8605],
        [-7.9943,-5.0266,-2.0579,-0.0786,1.9008,4.8693,7.8368]])

    # Measured on DEATH channels, 1->16; old DEATHs only -- 2016.04.14
    max_voltages = np.array([9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9,
                    9.5,9.5,9.5,9.5, 9.5,9.5,9.5,9.5, 9.5,9.5,9.5,9.5, 9.5,9.5,9.5,9.5])
    min_voltages = -max_voltages
    exact_voltages = np.array([-8, -5, -2, 0, 2, 5, 8])

    cal_gains_offsets = np.zeros([max_voltages.size, 2], dtype='uint32') # gains and offsets

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
        cal_gains_offsets[k] = [int(np.round(mb)), int(np.round(ob))]

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
            print("")
            # slope_correction = 32768*(1.0/lin_result.slope)
            # offset_correction = (65536//20)*(-10*lin_result.slope+lin_result.intercept)
            # offset_correction = (65536//20)*(-lin_result.intercept/lin_result.slope)

    # Output a text snippet suitable for copying'n'pasting into an Ionizer save file
    with open('death_outputs_to_use.txt', 'w') as cal_file:
        for k, vals in enumerate(cal_gains_offsets):
            cal_file.write("{DEATH channel %u gain} = {%u}, 0;\n" %(k, vals[0]))
            cal_file.write("{DEATH channel %u offset} = {%u}, 0;\n" %(k, vals[1]))

    # lin_offset_result = linregress(exact_voltages, cal_v_offset)


    # st()
