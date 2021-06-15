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
pos = numpy.array([-0.0004,0.0004])
freq = numpy.array([1.6e6,1.6e6])
offs = numpy.array([0.8,0.8])
linspace_fn=numpy.linspace

 
weights=[20000,18000,500,20000,500,100,0,200,0,0] # default weights used in the SolvePotentials2 routine (betas)
#solver2_weights=[   0,18000, 500,20000, 500,100,0,200,0,1e6], # offset relaxation
#solver2_weights=[50000,18000,1000,20000,1000,100,0, 20,0,0], #  HOA2 weights
#d_full=30e-3, #Mainz Voltag restriction off
#d_part=40e-3, #Mainz voltage restriction off
#d_full=10e-2,
#d_part=20e-2,
dfull=3e-3
dpart=6e-3
#d_full=5e-3,
#d_part=9e-3,

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
mu = wf_wdwtrans.raw_samples()




# plotting
ax= lambda nu: 2* (2 * np.pi * nu)**2 * (mass_Ca* atomic_mass_unit)/(2 * electron_charge)


sol_mus = mu
ys = []

dim =30 
xs = range(timesteps)

for i in range(dim):
    y = numpy.zeros(timesteps)
    
    for x in xs:
        y[x] = sol_mus[i,x]
    ys.append(y)

import matplotlib.pylab as plt

import matplotlib.pylab as plt
import matplotlib.ticker as mticker
from matplotlib import rc
import matplotlib

plt.figure(figsize=(6.3,7.9))

plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$V$'))
j = 7
i = j
k = 0

while k < 7:
    
    plt.plot(xs,ys[k],'--',label='L'+str(j-k))
    k = k+1
plt.plot(xs,ys[j],label='DCCa7')

while i < 14:
    i = i+1
    plt.plot(xs,ys[i],'--', label='R'+str(i-j))


plt.legend(bbox_to_anchor=(1.0,0.9))
plt.show()

# plotting objectives over time
mus = [] #contains mu vector for every time step


ts = range(timesteps)

for x in xs:
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
potax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f $V$'))
fieldax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f $\frac{V}{m}$'))
freqax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f $10^{6} \frac{V}{m^2}$'))


potax.plot(ts,wdwtrans.offsets,"--")
potax.plot(ts,offsetevol2)
fieldax.plot(ts,numpy.zeros(xs.size),"--")
fieldax.plot(ts,dfevol2)
freqax.plot(ts,ax(wdwtrans.freqs)*1e-6,"--")
freqax.plot(ts,numpy.array(sndderivevol2)*1e-6)

plt.show()
