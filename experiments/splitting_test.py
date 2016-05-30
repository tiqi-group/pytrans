#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

if __name__ == "__main__":

    analytic_plot = False
    test_split = True
    
    # Investigate a and b potentials
    if analytic_plot:
        a = -3
        b = 2

        x = np.linspace(-2,2,20000)
        y = a*x**2+b*x**4

        x_min = np.sqrt(-a/2/b)

        print(x_min)
        plt.plot(x,y)
        plt.plot(x, 2*np.abs(a)*(x-x_min)**2+(a*x_min**2+b*x_min**4))
        plt.plot(x, 2*np.abs(a)*(x+x_min)**2+(a*x_min**2+b*x_min**4))

        plt.show()

    if test_split:
        # plt.plot(trap_mom.transport_axis)
        roi_centre = 943//2
        roi_size = 20
        roi_idxes = range(roi_centre-roi_size//2,roi_centre+roi_size//2)

        roi_x = trap_mom.transport_axis[roi_idxes]

        a = 1*0.2e6
        b = 0*(1.5e6)**2
        y = a*roi_x**2 + b*roi_x**4-0.005
        
        # wd = WavDesired(y, roi_idxes, solver_weights={
        #     'r0_u_weights':np.zeros(30)})
        wd = WavDesiredWells([[0]],[[1.0*MHz]],[[-800*meV]],
                             desired_potential_params={'energy_threshold':100*meV})# ,
                             # solver_weights={
                             #     'r0':0,
                             #     # 'r1':0,
                             #     # 'r2':0,
                             #     'r0_u_ss':np.zeros(30)})
        ax = wd.plot(trap_mom.transport_axis)
        wf = Waveform(wd)
        st()
        wfp = calculate_potentials(trap_mom, wf)
        wfp.plot_one_wfm(0, ax)
        plt.show()
