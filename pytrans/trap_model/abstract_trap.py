#! python3

import numpy
from scipy.interpolate import splrep,splev
import pytrans
# dep - import piecePolyFit.py
# not in this file - import pickle
# a class that defines all the functions useful in a trap class

class AbstractTrap:
    
    Vmax = None
    Vmin = None
    start = None
    stop = None
    
    electrode_coords = None

    def xmid (self, bounds): 
        
        xm = []
        for b in bounds:
            xm.append( (b[0] + b[1])/2)
        return numpy.array(xm)

    def setup_xmids (self):

        if not hasattr(self, 'x_mids'):
            if hasattr(self, 'electrode_coords') and self.electrode_coords is not None:
                self.x_mids = self.xmid(self.electrode_coords)

    
    # purpose is the scaling of x_mid lists
    def scale_nested_list(self, inlist, scalefactor):
        ret_list = []
        for i in inlist:
            if isinstance(i, list):
                ret_list.append(self.scale_nested_list(i,scalefactor))
            elif isinstance(i, tuple):
                ret_list.append(tuple(scalefactor * j for j in i))
            else :
                ret_list.append(i * scalefactor) 
        return ret_list

                


    def setup_Vmaxmins (self):
        def V_to_list(V):
            if (isinstance(V,list)):
                assert v.size == self.numberofelectrodes, "dimension missmatch"
                return numpy.array(list)
            elif (isinstance(V,numpy.ndarray)):
                assert v.size == self.numberofelectrodes, "dimension missmatch"
                return V
            else:
                return numpy.array(self.numberofelectrodes * [V])
            
        if self.Vmax is None:
            raise ValueError('The maximum Voltage (Vmax) for the used Trap is not given')
        if self.Vmin is None:
            self.Vmin = self.Vmax
        self.Vmaxs = V_to_list(self.Vmax)
        self.Vmins = V_to_list(self.Vmin)
    

    # assumes transportaxis data is loaded
    # the smoothening s depends an the smoothness of tinterpoltaion data and avoids to close fitting, which makes the derivatives less useful. To large s will result in too lose fit 
    def setuptrapaxisBSpline(self,s = 3.2e-10):
        self.trapAxisSplineInterpol = [] # numpy.array([] * self.numberofelectrodes)
        for i in range(self.numberofelectrodes):
            # TODO check that the naming is equivalant between HOA2 and ETH
            self.trapAxisSplineInterpol.append(splrep(self.transport_axis,self.potentials[:,i],s= s,k=5))
    
    def FuncbyBspline(self,x,deriv):
        res = numpy.zeros(self.numberofelectrodes)
        for i in range(self.numberofelectrodes):
            res[i] = splev(x,self.trapAxisSplineInterpol[i],deriv)
        return res


    def overwriteGlobalVariables(self):
        pytrans.max_elec_voltage = self.Vmax
        pytrans.max_elec_voltages = self.Vmaxs
        pytrans.min_elec_voltage = self.Vmin
        pytrans.min_elec_voltages = self.Vmins
        pytrans.num_elecs = self.numberofelectrodes
        pytrans.default_elec_voltage = self.Vdefault
        pytrans.dac_channel_transform
        pytrans.dac_channel_transform = self.dac_channel_transform 
        pytrans.max_death_voltages = self.Vmaxs[self.dac_channel_transform]
        pytrans.min_death_voltages = self.Vmins[self.dac_channel_transform]



    # is called after all parameters are set. Can also becalled after changes or if the empty constructor was used
    def setup(self):
        self.setup_xmids()
        self.setup_Vmaxmins()
        self.overwriteGlobalVariables()

