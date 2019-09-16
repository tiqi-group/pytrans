#!/usr/bin/env python3
# static with ETH3d using solver1
# author: Sebastian Heinekamp

#imports
import numpy

import sys
sys.path.append("../")
from pytrans import *
from reorder import *
import copy as cp

# Loading conveyor stuff
import transport_utils as tu
import loading_utils as lu

# available Traps
from ETH3dTrap import ETH3dTrap as ETHMoments
from HOA2Trap import HOA2Trap as HOA2Moments


## Expremient settings
pos = -0.000
freq = 2000000
offset = 1

 
weights=[200000,18000,5000,20000,1000,100,0, 20,0,0] #  HOA2 weights
#d_full=30e-3, #Mainz Voltag restriction off
#d_part=40e-3, #Mainz voltage restriction off
#d_full=10e-2,
#d_part=20e-2,
dfull=3.8e-3
dpart=7e-3
#d_full=5e-3,
#d_part=9e-3,

#print(global_settings['USESOLVE2'])
global_settings['USESOLVE2'] = True 

trap = HOA2Moments()
trap.overwriteGlobalVariables() # makes sure the global variables correspond to the trap variables - only needed if not executed in Moments constructor

wdw2 = WavDesiredWells([pos],[freq],[offset],trap_m=trap,solver2_weights=weights, d_full=dfull, d_part=dpart)
wf = Waveform(wdw2)

mu2 = wf.raw_samples()
#print(mu2)

# calculacte errors
a = 2* (2 * np.pi * freq)**2 * (mass_Ca* atomic_mass_unit)/(2 * electron_charge)
print( "The offset error (normalized): ")
print((numpy.dot(mu2.T,trap.Func(pos,0)) - offset ) / offset)
print( "The first derivative at the well bottom")
print(numpy.dot(mu2.T, trap.Func(pos,1)))
print( "The frequency error (normalized): ")
print((numpy.dot(mu2.T, trap.Func(pos,2)) - a ) / a)



### Plotting
from scipy.interpolate import splrep,splev

import matplotlib.ticker as mticker
from matplotlib import rc
import matplotlib

#rc ('text', usetex=True)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode']=True

gridsize = ( 3,1)
plt.figure(figsize=(6.3, 7.9))
potax = plt.subplot2grid(gridsize, (0,0))
fieldax = plt.subplot2grid(gridsize, (1,0),sharex=potax)
freqax = plt.subplot2grid(gridsize, (2,0),sharex=potax)

potax.xaxis.set_major_locator(plt.MaxNLocator(7))
potax.xaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$\mu m$'))
potax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$V$'))
fieldax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$\frac{V}{m}$'))
freqax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.0f$10^{6}\frac{V}{m^2}$'))


fx = lambda mu, x, deriv: numpy.dot(mu.T, trap.Func(x,deriv))


xs = numpy.linspace(-0.1 * 10e-3,0.1 * 10e-3,400)
pot2xs = numpy.array(list(map(lambda x: fx(mu2,x,0),xs)))
dpot2xs = numpy.array(list(map(lambda x: fx(mu2,x,1),xs)))
d2pot2xs = numpy.array(list(map(lambda x: fx(mu2,x,2),xs)))



i = 0

potax.plot(xs*1e6,pot2xs,label='HOA2')

potaxright = potax.twinx()
potax.plot(xs*1e6,numpy.array([offset] * 400),'--',dashes=(5,20),color='grey',linewidth=0.8)
potax.axvline(x=pos,linestyle='--',dashes=(5,20),color='grey',linewidth=0.8)
potaxright.set_yticks([offset])
#potaxright.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'$\Phi_0$'))
potaxright.yaxis.set_major_locator(mticker.FixedLocator([offset]))
potaxright.yaxis.set_major_formatter(mticker.FixedFormatter([r'$\Phi_0$']))
potaxright.set_ylim(potax.get_ylim()[0],potax.get_ylim()[-1])



#plt.ylim([-0.001,0.001])
fieldax.plot(xs*1e6,dpot2xs)

fieldaxright = fieldax.twinx()
fieldax.plot(xs*1e6,numpy.array([0.0] * 400),'--',dashes=(5,20),color='grey',linewidth=0.8)
fieldax.axvline(x=pos,linestyle='--',dashes=(5,20),color='grey',linewidth=0.8)
fieldaxright.set_yticks([0.0])
#fieldaxright.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'$\Phi_0$'))
fieldaxright.yaxis.set_major_locator(mticker.FixedLocator([0.0]))
fieldaxright.yaxis.set_major_formatter(mticker.FixedFormatter([r'']))
fieldaxright.set_ylim(fieldax.get_ylim()[0],fieldax.get_ylim()[-1])

freqax.plot(xs*1e6,d2pot2xs*1e-6)

freqaxright = freqax.twinx()
freqax.plot(xs*1e6,numpy.array([a*1e-6] * 400),'--',dashes=(5,20),color='grey',linewidth=0.8)
freqax.axvline(x=pos,linestyle='--',dashes=(5,20),color='grey',linewidth=0.8)
freqaxright.set_yticks([a*1e-6])
#freqaxright.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'$\Phi_0$'))
freqaxright.yaxis.set_major_locator(mticker.FixedLocator([a*1e-6]))
freqaxright.yaxis.set_major_formatter(mticker.FixedFormatter([r'$a$']))
freqaxright.set_ylim(freqax.get_ylim()[0],freqax.get_ylim()[-1])

potax.legend()
#plt.show()
plt.savefig('png/static_HOA2.png')
