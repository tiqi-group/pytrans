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
timesteps = 400
pos = numpy.array([-0.0004,0.000])
freq = numpy.array([1.6e6,1.6e6])
offs = numpy.array([0.8,0.8])
linspace_fn=numpy.linspace

 
weights=[20000,15000,500,1000000,10000,100,0,200,0,0] # default weights used in the SolvePotentials2 routine (betas)
#solver2_weights=[   0,18000, 500,20000, 500,100,0,200,0,1e6], # offset relaxation
#solver2_weights=[50000,18000,1000,20000,1000,100,0, 20,0,0], #  HOA2 weights
dfull=30e-2 #Mainz Voltag restriction off
dpart=40e-2 #Mainz voltage restriction off
#dfull=10e-3
#dpart=20e-3
#dfull=3.8e-3
#dpart=7e-3
#dfull=7e-3
#dpart=11e-3

global_settings['USESOLVE2'] = True

trap = ETHMoments()
trap.overwriteGlobalVariables() # makes sure the global variables correspond to the trap variables - only needed if not executed in Moments constructor

wdwtrans = WavDesiredWells(
            [linspace_fn(pos[0], pos[1], timesteps)],
            [linspace_fn(freq[0], freq[1], timesteps)],
            [linspace_fn(offs[0], offs[1], timesteps)],
            force_static_ends=False,
            desc=" {:.3f}->{:.3f} MHz, {:.1f}->{:.1f} meV".format(freq[0], freq[1], offs[0], offs[1]),
            trap_m=trap,
            solver2_weights=weights,
            d_full=dfull,
            d_part=dpart
            )

wf_wdwtrans = Waveform(wdwtrans) 
mu2 = wf_wdwtrans.raw_samples()




# plotting
ax= lambda nu: 2* (2 * np.pi * nu)**2 * (mass_Ca* atomic_mass_unit)/(2 * electron_charge)


sol_mus = mu2
ys = []

dim =30 
xss = range(timesteps)

for i in range(dim):
    y = numpy.zeros(timesteps)
    
    for x in xss:
        y[x] = sol_mus[i,x]
    ys.append(y)

import matplotlib.pylab as plt

import matplotlib.pylab as plt
import matplotlib.ticker as mticker
from matplotlib import rc
import matplotlib

plt.figure(figsize=(12,6))

plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f $V$'))
for y in ys:
    plt.plot(xss,y,'--')

plt.show()

# calculating with solver1 for comparison
global_settings['USESOLVE2'] = False

wf_wdwtrans = Waveform(wdwtrans) 
mu1 = wf_wdwtrans.raw_samples()

# plotting objectives over time
mus = [] #contains mu vector for every time step


ts = range(timesteps)

for x in xss:
    mul = numpy.zeros(dim)

    mul = sol_mus[:,x]
    mus.append(mul)
        
xs = wdwtrans.positions

pot0= lambda x,mul: numpy.dot(mul.T , trap.Func( x,0) )
pot1= lambda x,mul: numpy.dot(mul.T , trap.Func( x,1) )
pot2= lambda x,mul: numpy.dot(mul.T , trap.Func( x,2) )

offsetevol2 = []
dfevol2 = []
sndderivevol2 =[]

for x,mu in zip(xs,mus):
    offsetevol2.append(numpy.asarray(pot0(x,mu)).reshape(-1))
    dfevol2.append(numpy.asarray(pot1(x,mu)).reshape(-1))
    sndderivevol2.append(numpy.asarray(pot2(x,mu)).reshape(-1))


mus = [] #contains mu vector for every time step
sol_mus = mu1

ts = range(timesteps)

for x in xss:
    mul = numpy.zeros(dim)

    mul = sol_mus[:,x]
    mus.append(mul)
        
offsetevol1 = []
dfevol1 = []
sndderivevol1 =[]

for x,mu in zip(xs,mus):
    offsetevol1.append(numpy.asarray(pot0(x,mu)).reshape(-1))
    dfevol1.append(numpy.asarray(pot1(x,mu)).reshape(-1))
    sndderivevol1.append(numpy.asarray(pot2(x,mu)).reshape(-1))
           


import matplotlib.pylab as plt
import matplotlib.ticker as mticker
from matplotlib import rc
import matplotlib

#rc ('text', usetex=True)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode']=True

gridsize = ( 3,1)
plt.figure(figsize=(12, 20 ))
potax = plt.subplot2grid(gridsize, (0,0))
fieldax = plt.subplot2grid(gridsize, (1,0),sharex=potax)
freqax = plt.subplot2grid(gridsize, (2,0),sharex=potax)

#potax.xaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f $\mu m$'))
potax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.3f $V$'))
fieldax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f $\frac{V}{m}$'))
freqax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f $10^{6} \frac{V}{m^2}$'))


potax.plot(ts,wdwtrans.offsets,"--")
potax.plot(ts,offsetevol2)
potax.plot(ts,offsetevol1)
fieldax.plot(ts,numpy.zeros(xs.size),"--")
fieldax.plot(ts,dfevol2)
fieldax.plot(ts,dfevol1)
freqax.plot(ts,ax(wdwtrans.freqs)*1e-6,"--")
freqax.plot(ts,numpy.array(sndderivevol2)*1e-6)
freqax.plot(ts,numpy.array(sndderivevol1)*1e-6)

plt.show()

