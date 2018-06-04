 #! python3

from AbstractTrap import AbstractTrap as ATrap
import PotFuncProvider
import numpy as np

# implementation for the HOA2 Trap

class HOA2Trap(ATrap):

    # only man axis 

    # #Trap attributes

    
 
    Vmax = 19.9
    Vmin = - Vmax

    numberofelectrodes = 76
    max_slew_rate = 5 / 10-6
    
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
            moments_path = os.poth.join(os.path.dirname(__file__),"moments_file","HOA2.pickle"),
            potentials_path = os.poth.join(os.path.dirname(__file__),"moments_file","HOA2.pickle")
            ):

        # scale xs to mu-meter and set x_mids
        self.x_mids = self.scale_nested_list(self.xs,10e-6)

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

        with open(self.pickle_file,'rb') as f:
            pfile = pickle.load(f)

        pot3d = potentials_3d()
        pot3d.potentials = pfile[0]
        pot3d.origin = pfile[1]
        pot3d.spacing = pfile[2]
        pot3d.dimensions = pfile[3]
        pot3d.x = pfile[4]
        pot3d.y = pfile[5]
        pot3d.z = pfile[6]
        pot3d.nx = np.shape(x)[0]
        pot3d.ny = np.shape(y)[0]
        pot3d.nz = np.shape(z)[0]
        pot3d.ntot = pot3d.nx * pot3d.ny * pot3d.nz # total number of points in mesh
        pot3d.xx = pfile[7] # vector with the x coordinates for all the mesh points, flattened
        pot3d.yy = pfile[8] # i.e. potentials['ElectrodeName'][ind] = V(xx[ind],yy[ind],zz[ind])
        pot3d.zz = pfile[9]
        pot3d.coordinates = pfile[10] # = [xx, yy, zz]
        pot3d.fit_coord3d = np.column_stack( (xx**2, yy**2, zz**2, xx*yy, xx*zz, yy*zz, xx, yy, zz, np.ones_like(zz)) ) # used for finding potential eigenaxes in 3d
        zz2d, yy2d = np.meshgrid(z,y) # coordinates for one slice in the radial plane
        yy2d = yy2d.flatten(order='F')
        zz2d = zz2d.flatten(order='F')
        pot3d.yy2d = yy2d
        pot3d.zz2d = zz2d
        pot3d.fit_coord2d = np.column_stack( (yy2d**2, zz2d**2, yy2d*zz2d, yy2d, zz2d, np.ones_like(zz2d)) ) # used for finding potential eigenaxes in 2d
        
        # TODO shim segments combination

        self.pot3d = pot3d

    def load_trap_axis_potential_data(self, moments_path):
        if self.pot3d is None: # since we extract the trap axis from the same pickle file it makes sens to first load the full 3d file
            self.load_3d_potential_data(moments_path)
        def calculate_slice(pot3d):
            return slice(pot3d.ny * floor(pot3d.nz/2) + floor(pot3d.ny /2),None,pot3d.ny * pot3d.nz)
        slice = calculate_slice(self.pot3d) #points to the entries that contain the trap axis potentials
        # TODO to ensure backwards compatability
        # Problem for now is that there is nothing comparible for now (could use spline inter polation and its results
        #self.electrode_moments = []
            
        
        self.potentals = np.zeros(self.pot3d.nx,numberofelectrodes)
        for k in range(numberofelectrodes)
            self.potentials[:,k] = self.potentials[self.electrode_names(k)][slice]
        
        # other properties, that ensure the class is compatible to the Moments class
        self.transport_axis = self.pot3d.x
        self.rf_pondpot = self.pot3d.potentials['RF_pondpot_1V1MHz1amu']

        # TODO to ensure backwards compatability
        # Problem for now is that there is nothing comparible for now (could use spline inter polation and its results
        #self.shim_moments = []


    
    # this allows to hide the actually used interpolation behind this interface funcion (e.g. analytic solutions) and still reuse the same interpolation implementations between different Traps 
    def Func(x,deriv):
        return FunkbyBspline(x,deriv)




