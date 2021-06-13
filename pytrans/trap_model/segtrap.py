#! python3

from .abstract_trap import AbstractTrap
import numpy
import os
import scipy.io as sio
import pickle
import warnings
from pytrans.constants import *

# this class impelments the AbstractTrap for the ETH3dTrap
# also intents to serve as a documenting sample implementation. (other options are explain in the comments)


class ETH3dTrap(AbstractTrap):

    data_path = '/home/carmelo/eth/pytrans/moments_data/segtrap'
    moments_path = os.path.join(data_path, "DanielTrapMomentsTransport.mat")
    potential_path = os.path.join(data_path, "trap.pickle")

    num_electrodes = 30  # they come in pairs
    default_V = 5
    min_V = -10
    max_V = 10

    def __str__(self):
        return "ETH3dTrap"
    # #Trap attributes

    def __init__(self):
        super().__init__()

        self.load_trap_axis_potential_data()

    def load_trap_axis_potential_data(self):
        """ Based on reduced_data_ludwig.m, reconstructed here.
        Extracts and stores the potentials along the trap axis due to the various electrodes,
        as well as the first few spatial derivatives with respect to the trap axis. """

        data = sio.loadmat(self.moments_path, struct_as_record=False)[
            'DATA'][0][0]

        starting_shim_electrode = 30
        num_electrodes = self.num_electrodes  # Control electrodes DCCa0 to DCCa14 and DCCb0 to DCCb14
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
        # Potential due to RF electrodes along trap axis. Needs to be scaled with rf freq, voltage and ion mass.
        self.rf_pondpot = data.RF_pondpot

        # More complete potential data
        # Organised as (number of z locations) * (number of electrodes) (different from Matlab)
        # self.moments = numpy.zeros([len(self.transport_axis), num_electrodes])
        # for k in range(num_electrodes):
        #     self.moments[:, k] = self.electrode_moments[k][:, 0]
        
        # ### transpose
        self.moments = numpy.zeros([num_electrodes, len(self.transport_axis)])
        for k in range(num_electrodes):
            self.moments[k] = self.electrode_moments[k][:, 0]

        # Higher-res potential data [don't need for now]

    def load_3d_potential_data(self, potential_path):
        """ Loads the 3d potentials due to the individual trap electrodes as
        obtained from simulations performed with the NIST BEM software.
        This data is primarily used to calculate the radial frequencies
        and principal axes within the trap.
        """

        with open(potential_path, 'rb') as f:
            potentials, origin, spacing, dimensions, x, y, z, xx, yy, zz, coordinates = pickle.load(
                f)

        # Add up the contributions of the shim segments and add to dictionary
        V_DCsa = potentials['DCSa1'] + potentials['DCSa2'] + \
            potentials['DCSa3'] + potentials['DCSa4'] + potentials['DCSa5']
        V_DCsb = potentials['DCSb1'] + potentials['DCSb2'] + \
            potentials['DCSb3'] + potentials['DCSb4'] + potentials['DCSb5']
        V_DCsc = potentials['DCSc1'] + potentials['DCSc2'] + \
            potentials['DCSc3'] + potentials['DCSc4'] + potentials['DCSc5']
        V_DCsd = potentials['DCSd1'] + potentials['DCSd2'] + \
            potentials['DCSd3'] + potentials['DCSd4'] + potentials['DCSd5']

        potentials.update({'DCSa': V_DCsa})
        potentials.update({'DCSb': V_DCsb})
        potentials.update({'DCSc': V_DCsc})
        potentials.update({'DCSd': V_DCsd})

        # Define dummy class to use similar to a C struct in order to
        # bundle the 3d potential data into a single object.
        class potentials_3d:
            pass

        pot3d = potentials_3d()
        # dictionary containing the potentials of all the control & shim & rf electrodes
        pot3d.potentials = potentials
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
        # i.e. potentials['ElectrodeName'][ind] = V(xx[ind],yy[ind],zz[ind])
        pot3d.yy = yy
        pot3d.zz = zz
        pot3d.coordinates = coordinates  # = [xx, yy, zz]
        # used for finding potential eigenaxes in 3d
        pot3d.fit_coord3d = numpy.column_stack(
            (xx**2, yy**2, zz**2, xx * yy, xx * zz, yy * zz, xx, yy, zz, numpy.ones_like(zz)))
        # coordinates for one slice in the radial plane
        zz2d, yy2d = numpy.meshgrid(z, y)
        yy2d = yy2d.flatten(order='F')
        zz2d = zz2d.flatten(order='F')
        pot3d.yy2d = yy2d
        pot3d.zz2d = zz2d
        # used for finding potential eigenaxes in 2d
        pot3d.fit_coord2d = numpy.column_stack(
            (yy2d**2, zz2d**2, yy2d * zz2d, yy2d, zz2d, numpy.ones_like(zz2d)))
        self.pot3d = pot3d

    # this allows to hide the actually used interpolation behind this interface funcion (e.g. analytic solutions) and still reuse the same interpolation implementations between different Traps
    def Func(self, x, deriv):
        return self.FuncbyBspline(x, deriv)
