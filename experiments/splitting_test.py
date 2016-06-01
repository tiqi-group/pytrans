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
        # wd = WavDesiredWells([[-850*um],[0*um], [850*um]],[[1.0*MHz],[1.3*MHz], [1.5*MHz]],[[-800*meV],[0*meV], [-840*meV]],
        #                      desired_potential_params={'energy_threshold':100*meV})
        steps = 10
        
        wd = WavDesiredWells([np.linspace(-500,-300,steps)*um,np.linspace(300,500,steps)*um],
                             [np.linspace(1,1.5,steps)*MHz,np.linspace(1.5,1,steps)*MHz],
                             [np.linspace(-800,100,steps)*meV,np.linspace(-500,0,steps)*meV],
                             
                             desired_potential_params={'energy_threshold':100*meV},
                             solver_weights={'r1':0, 'r2':0})
                
        wf = Waveform(wd)
        wfp = calculate_potentials(trap_mom, wf)

        # Generates a lot of figures
        ax = wd.plot(0, trap_mom.transport_axis)
        wfp.plot_one_wfm(0, ax)
        for k in range(1,steps):
            wd.plot(k, trap_mom.transport_axis, ax)
            wfp.plot_one_wfm(k, ax)

        plt.show()
        print(wfp.find_wells(0,'precise'))
