#! python3

from .abstract_trap import _old_AbstractTrap as ATrap
import numpy
import os
import scipy.io as sio
import pickle
import warnings
from pytrans.units import *

# this class impelments the AbstractTrap for the ETH3dTrap
# also intents to serve as a documenting sample implementation. (other options are explain in the comments)

data_path = '/home/carmelo/ETH/pytrans/moments_data/segtrap'

moments_path = os.path.join(data_path, "DanielTrapMomentsTransport.mat")
potential_path = os.path.join(data_path, "trap.pickle")


class ETH3dTrap(ATrap):

    def __str__(self):
        return "ETH3dTrap"
    # #Trap attributes

    Vmax = 8.9
    Vmin = - Vmax

    # Vdefault = 5
    Vdefault = 0

    numberofelectrodes = 30

    electrode_coords = 1e-6 * numpy.array([[-3000, -2545], [-2535, -1535], [-1515, -1015], [-995, -695], [-675, -520], [-500, -345], [-325, -170], [-150, 150], [170, 325], [345, 500], [520, 675], [695, 995], [1015, 1515], [1535, 2535], [2545, 3000], [-3000, -2545], [-2535, -1535], [-1515, -1015], [-995, -695], [-675, -520], [-500, -345], [-325, -170], [-150, 150], [170, 325], [345, 500], [520, 675], [695, 995], [1015, 1515], [1535, 2535], [2545, 3000]])

    symmetry = [0] * numberofelectrodes
    symmetry[:15] = range(numberofelectrodes)[15:]
    symmetry[15:] = range(numberofelectrodes)[:15]

    # Values represent indices of which electrode each DEATH output
    # drives, from 0->31. E.g. dac_channel_transform[5] = 1 tells us that
    # DEATH output 5 drives Electrode 1.
    dac_channel_transform = numpy.array([0, 15, 3, 18, 1, 16, 4, 19,   2, 17, 5, 20, -7, 14, 6, 21,
                                         11, 26, 7, 22, 12, 27, 8, 23,  13, 28, 9, 24, -22, 29, 10, 25])

    # This array is written to geometrically show which electrodes
    # are controlled by which DEATH channels.
    # Grouping of 4, 3, 1, 3, 4 refers to load, split, exp, split, load.
    physical_electrode_transform = numpy.array([0, 4, 8, 2,  6, 10, 14,  18,  22, 26, 30,  16, 20, 24, 13,
                                                1, 5, 9, 3,  7, 11, 15,  19,  23, 27, 31,  17, 21, 25, 29])

    def electrode_names(self, x):
        return ('DCC' + ('c' + str(x) if x < 15 else 'a' + str(x - 15)))

    def __init__(self, moments_path=moments_path, potential_path=potential_path):

        # sets up the Vmin&Vmax Vector and calculates necessary trap geometry
        self.setup()

        self.load_trap_axis_potential_data(moments_path)
        self.load_3d_potential_data(potential_path)

        # additions for solver2
        self.max_slew_rate = 5 / us  # (units of volts / s, quarter of DEATH AD8021 op-amps)

        self.setuptrapaxisBSpline(s=3.2e-10)

    def load_trap_axis_potential_data(self, moments_path):
        """ Based on reduced_data_ludwig.m, reconstructed here.
        Extracts and stores the potentials along the trap axis due to the various electrodes,
        as well as the first few spatial derivatives with respect to the trap axis. """

        data = sio.loadmat(moments_path, struct_as_record=False)['DATA'][0][0]

        starting_shim_electrode = 30
        num_electrodes = 30  # Control electrodes DCCa0 to DCCa14 and DCCb0 to DCCb14
        num_shims = 20  # Shim electrodes DCS[a,b,c,d][1,2,3,4,5] e.g. DCSa1

        # The electrode moments store the potential of the respective electrode
        # along the trap axis, as well as the first few derivatives. E.g.
        # V(z) = electrode_moments[:,0]
        # V'(z) = electrode_moments[:,1]
        # etc. up to V(5)(z) = electrode_moments[:,5]
        # However, for V'''(z) and higher derivatives the data becomes increasingly noisy.
        self.electrode_moments = []
        self.shim_moments = []

        for q in range(num_electrodes):
            self.electrode_moments.append(data.electrode[0, q].moments)

        for q in range(starting_shim_electrode, num_shims + starting_shim_electrode):
            self.shim_moments.append(data.electrode[0, q].moments)

        self.transport_axis = data.transport_axis.flatten()
        self.rf_pondpot = data.RF_pondpot  # Potential due to RF electrodes along trap axis. Needs to be scaled with rf freq, voltage and ion mass.

        # More complete potential data
        # Organised as (number of z locations) * (number of electrodes) (different from Matlab)
        self.potentials = numpy.zeros([len(self.transport_axis), num_electrodes])
        for k in range(num_electrodes):
            self.potentials[:, k] = self.electrode_moments[k][:, 0]

        # Higher-res potential data [don't need for now]

    def load_3d_potential_data(self, potential_path):
        """ Loads the 3d potentials due to the individual trap electrodes as
        obtained from simulations performed with the NIST BEM software.
        This data is primarily used to calculate the radial frequencies
        and principal axes within the trap.
        """

        with open(potential_path, 'rb') as f:
            potentials, origin, spacing, dimensions, x, y, z, xx, yy, zz, coordinates = pickle.load(f)

        # Add up the contributions of the shim segments and add to dictionary
        V_DCsa = potentials['DCSa1'] + potentials['DCSa2'] + potentials['DCSa3'] + potentials['DCSa4'] + potentials['DCSa5']
        V_DCsb = potentials['DCSb1'] + potentials['DCSb2'] + potentials['DCSb3'] + potentials['DCSb4'] + potentials['DCSb5']
        V_DCsc = potentials['DCSc1'] + potentials['DCSc2'] + potentials['DCSc3'] + potentials['DCSc4'] + potentials['DCSc5']
        V_DCsd = potentials['DCSd1'] + potentials['DCSd2'] + potentials['DCSd3'] + potentials['DCSd4'] + potentials['DCSd5']

        potentials.update({'DCSa': V_DCsa})
        potentials.update({'DCSb': V_DCsb})
        potentials.update({'DCSc': V_DCsc})
        potentials.update({'DCSd': V_DCsd})

        # Define dummy class to use similar to a C struct in order to
        # bundle the 3d potential data into a single object.
        class potentials_3d:
            pass

        pot3d = potentials_3d()
        pot3d.potentials = potentials  # dictionary containing the potentials of all the control & shim & rf electrodes
        pot3d.origin = origin  # origin of the mesh
        pot3d.spacing = spacing  # spacing of the mesh along the various axes
        pot3d.dimensions = dimensions  # number of points in the mesh along the various axes
        pot3d.x = x  # vector containing the points along a single axis
        pot3d.y = y  # i.e. y = [-11, -9, ..., 9, 11]*um
        pot3d.z = z
        pot3d.nx = numpy.shape(x)[0]
        pot3d.ny = numpy.shape(y)[0]
        pot3d.nz = numpy.shape(z)[0]
        pot3d.ntot = pot3d.nx * pot3d.ny * pot3d.nz  # total number of points in mesh
        pot3d.xx = xx  # vector with the x coordinates for all the mesh points, flattened
        pot3d.yy = yy  # i.e. potentials['ElectrodeName'][ind] = V(xx[ind],yy[ind],zz[ind])
        pot3d.zz = zz
        pot3d.coordinates = coordinates  # = [xx, yy, zz]
        pot3d.fit_coord3d = numpy.column_stack((xx**2, yy**2, zz**2, xx * yy, xx * zz, yy * zz, xx, yy, zz, numpy.ones_like(zz)))  # used for finding potential eigenaxes in 3d
        zz2d, yy2d = numpy.meshgrid(z, y)  # coordinates for one slice in the radial plane
        yy2d = yy2d.flatten(order='F')
        zz2d = zz2d.flatten(order='F')
        pot3d.yy2d = yy2d
        pot3d.zz2d = zz2d
        pot3d.fit_coord2d = numpy.column_stack((yy2d**2, zz2d**2, yy2d * zz2d, yy2d, zz2d, numpy.ones_like(zz2d)))  # used for finding potential eigenaxes in 2d
        self.pot3d = pot3d

    # this allows to hide the actually used interpolation behind this interface funcion (e.g. analytic solutions) and still reuse the same interpolation implementations between different Traps
    def Func(self, x, deriv):
        return self.FuncbyBspline(x, deriv)
