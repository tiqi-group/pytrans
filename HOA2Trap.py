 #! python3

from AbstractTrap import AbstractTrap as ATrap
import numpy as np
import os
import pickle
import math
from units import *

# implementation for the HOA2 Trap

class HOA2Trap(ATrap):

    def __str__(self):
        return "HOA2Trap"

    # only man axis 

    # #Trap attributes

    Vdefault = 6 
 
    Vmax = 8.9
    Vmin = - Vmax

    numberofelectrodes = 76
    max_slew_rate = 5 / 10-6

    # NOTE - depends on exerimental setup
    # NOTE - No real data so far
    # Values represent indices of which electrode each DEATH output
    # drives, from 0->31. E.g. dac_channel_transform[5] = 1 tells us that
    # DEATH output 5 drives Electrode 1.
    dac_channel_transform = list(range(76)) + [0,1]
 
 
    # This array is written to geometrically show which electrodes
    # are controlled by which DEATH channels.
    # Grouping of 4, 3, 1, 3, 4 refers to load, split, exp, split, load.
    physical_electrode_transform = list(range(76))
    
    # instead of calculated locations
    xs = list(range(-630,631,70)) * 2# Q
    xs.sort()
    xs += [[-770,770,-1050,1050,-1330,1330,-1610,1610]] #G01
    xs += [[-770,770,-1050,1050,-1330,1330,-1610,1610]] #G02
    xs += [[-700,700,-980,980,-1260,1260,1540,-1540]] #G03
    xs += [[-700,700,-980,980,-1260,1260,1540,-1540]] #G04
    xs += [[-910,910,-1190,1190,-1470,1470]] #G05
    xs += [[-910,910,-1190,1190,-1470,1470]] #G06
    xs += [[-840,840,-1120,1120,-1400,1400]] #G07
    xs += [[-840,840,-1120,1120,-1400,1400]] #G08
    xs += [[1880,-1880]] #T1
    xs += [[1880,-1880]] #T2
    xs += [[1782,-1782]] #T3
    xs += [[1782,-1782]] #T4
    xs += [[-1692,1692]] #T5
    xs += [[-1692,1692]] #T6
    xs += 16 * [100000] # basically sets everything in the legs to zero 
    xs += 2 * [[-2259.1,2259.1]] #Y17 and Y18
    xs += 2 * [[-2163.9,2163.9]] #Y19 and Y20
    xs += 2 * [[-2069.6,2069.6]] #Y21 and Y22
    xs += 2 * [[-1975.2,1975.2]] #Y17 and Y18
    # L00 ignored 
    
    # needed to enforce the absolut symmetry
    symmetry = [0] * numberofelectrodes
    symmetry[::2] = range(numberofelectrodes)[1::2] 
    symmetry[1::2] = range(numberofelectrodes)[::2] 

    pot3d = None
    

    def electrode_names(self,x):
        # 0 bis 37 -> Q
        if x < 38:
            return 'Q'+ str(x +1).zfill(2)
        # 38 bis 45-> G
        elif x<46:
            return 'G'+ str(x - 37).zfill(2)
        # 46 bis 51 -> T
        elif x<52: 
            return 'T'+ str(x - 45).zfill(2)
        # 52 bis 75 -> Y
        else:
            return 'Y' + str(x - 51).zfill(2)
    
    def __init__(self,
            moments_path = os.path.join(os.path.dirname(__file__),"moments_file","HOA2.pickle"),
            potentials_path = os.path.join(os.path.dirname(__file__),"moments_file","HOA2.pickle")
            ):

        # scale xs to mu-meter and set x_mids
        self.x_mids = self.scale_nested_list(self.xs,10e-7)

        # sets up the Vmin&Vmax Vector and calculates necessary trap geometry
        self.setup()
    
        self.load_trap_axis_potential_data(moments_path) # also runs the load_3d_potential_data
        # the smoothening s depends an the smoothness of tinterpoltaion data and avoids to close fitting, which makes the derivatives less useful. To large s will result in too lose fit 
        
        self.setuptrapaxisBSpline(s=3.2e-10)
        
        # additions for solver2
        self.max_slew_rate = 5 / us # (units of volts / s, quarter of DEATH AD8021 op-amps)
        


    def load_3d_potential_data(self,potential_path):
        class potentials_3d:
            pass

        with open(potential_path,'rb') as f:
            pfile = pickle.load(f)

        pot3d = potentials_3d()
        pot3d.potentials = pfile[0]
        pot3d.origin = pfile[1]
        pot3d.spacing = pfile[2]
        pot3d.dimensions = pfile[3]
        pot3d.x = pfile[4]
        pot3d.y = pfile[5]
        pot3d.z = pfile[6]
        pot3d.nx = np.shape(pot3d.x)[0]
        pot3d.ny = np.shape(pot3d.y)[0]
        pot3d.nz = np.shape(pot3d.z)[0]
        pot3d.ntot = pot3d.nx * pot3d.ny * pot3d.nz # total number of points in mesh
        pot3d.xx = pfile[7] # vector with the x coordinates for all the mesh points, flattened
        pot3d.yy = pfile[8] # i.e. potentials['ElectrodeName'][ind] = V(xx[ind],yy[ind],zz[ind])
        pot3d.zz = pfile[9]
        pot3d.coordinates = pfile[10] # = [xx, yy, zz]
        pot3d.fit_coord3d = np.column_stack( (pot3d.xx**2, pot3d.yy**2, pot3d.zz**2, pot3d.xx*pot3d.yy, pot3d.xx*pot3d.zz, pot3d.yy*pot3d.zz, pot3d.xx, pot3d.yy, pot3d.zz, np.ones_like(pot3d.zz)) ) # used for finding potential eigenaxes in 3d
        zz2d, yy2d = np.meshgrid(pot3d.z,pot3d.y) # coordinates for one slice in the radial plane
        yy2d = yy2d.flatten(order='F')
        zz2d = zz2d.flatten(order='F')
        pot3d.yy2d = yy2d
        pot3d.zz2d = zz2d
        pot3d.fit_coord2d = np.column_stack( (pot3d.yy2d**2, pot3d.zz2d**2, pot3d.yy2d*pot3d.zz2d, pot3d.yy2d, pot3d.zz2d, np.ones_like(pot3d.zz2d)) ) # used for finding potential eigenaxes in 2d
        
        # NOTE shim segments combination missing

        self.pot3d = pot3d

    def load_trap_axis_potential_data(self, moments_path):
        if self.pot3d is None: # since we extract the trap axis from the same pickle file it makes sens to first load the full 3d file
            self.load_3d_potential_data(moments_path)
        def calculate_slice(pot3d):
            return slice(pot3d.ny * math.floor(pot3d.nz/2) + math.floor(pot3d.ny /2),None,pot3d.ny * pot3d.nz)
        s = calculate_slice(self.pot3d) #points to the entries that contain the trap axis potentials
        
        # Problem for now is that there is nothing comparible for now (could use spline inter polation and its results
        #self.electrode_moments = []
        
        self.potentials = np.zeros((self.pot3d.nx,self.numberofelectrodes))
        for k in range(self.numberofelectrodes):
            self.potentials[:,k] = self.pot3d.potentials[self.electrode_names(k)][s]
        
        # other properties, that ensure the class is compatible to the Moments class
        self.transport_axis = self.pot3d.x
        self.rf_pondpot = self.pot3d.potentials['RF_pondpot_1V1MHz1amu']

        #self.shim_moments = []

    # this allows to hide the actually used interpolation behind this interface funcion (e.g. analytic solutions) and still reuse the same interpolation implementations between different Traps 
    def Func(self,x,deriv):
        return self.FuncbyBspline(x,deriv)

