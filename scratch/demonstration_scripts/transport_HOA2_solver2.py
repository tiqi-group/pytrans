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

from scipy.interpolate import splrep,splev

import matplotlib.ticker as mticker
from matplotlib import rc
import matplotlib


## Expremient settings
timesteps = 400
pos = numpy.array([-0.0004,0.0004])
freq = numpy.array([1.6e6,1.6e6])
offs = numpy.array([0.8,0.8])
linspace_fn=numpy.linspace

 
weights=[50000,18000,1000,20000000,1000,100,0, 20,0,0] #  HOA2 weights
dfull=3.8e-3
dpart=7e-3

global_settings['USESOLVE2'] = True

trap = HOA2Moments()
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

dim =76 
xs = range(timesteps)

for i in range(dim):
    y = numpy.zeros(timesteps)
    
    for x in xs:
        y[x] = sol_mus[i,x]
    ys.append(y)

import matplotlib.pylab as plt


f = plt.figure(figsize=(10,5))
f.subplots_adjust(left=0.15)
f.subplots_adjust(right=0.92)
axi = f.add_subplot(111)
j = 19 
i = j
k = 5 

while k < 19:
    
    axi.plot(xs,ys[k],'--',label='L'+str(int((j-k)/2)))
    k = k+2
axi.plot(xs,ys[j],label='Q20')

while i < 32:
    i = i+2
    axi.plot(xs,ys[i],'--', label='R'+str(int((i-j)/2)))


axi.legend(bbox_to_anchor=(-0.05,1.0))

axi.set_ylim(0,trap.Vmax+0.5)

rightx = axi.twinx()
rightx.set_yticks([trap.Vdefault,trap.Vmax])
rightx.yaxis.set_major_locator(mticker.FixedLocator([trap.Vdefault,trap.Vmax]))
rightx.yaxis.set_major_formatter(mticker.FixedFormatter([r'$V_{default}$',r'$V_{max}$']))
rightx.set_ylim(axi.get_ylim()[0],axi.get_ylim()[-1])

axi.plot(numpy.array(xs),numpy.multiply(numpy.ones(timesteps),trap.Vmax),"--",dashes=(5,20),color='grey',linewidth=0.8)

axi.set_xlabel('timesteps')
axi.set_ylabel('voltage applied to electrode [V]')

plt.savefig('png/transport_HOA2_electrode_evol.png')


mus = [] #contains mu vector for every time step

timesteps =  400 # how many timesteps
#dim = 30 # How many electrodes
dim = 76
xs = range(timesteps)

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

#plt.figure()
f, axis = plt.subplots(3,1,sharex=True,figsize=(6.3,7.9))
f.subplots_adjust(left=0.15)

axis[0].xaxis.set_major_locator(plt.MaxNLocator(7))
axis[0].xaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$\mu m$'))
axis[0].yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$V$'))
axis[1].yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.1f $10^{-7}\frac{V}{m}$'))
axis[2].yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.0f $10^{6}\frac{V}{m^2}$'))

axis[0].plot(xs*1e6,wdwtrans.offsets,"--",dashes=(5,20),color='grey',linewidth=0.8)
#axis[0].plot(xs,offsetevol1)
axis[0].plot(xs*1e6,offsetevol2)

axis[0].set_ylim(0.6,1)

ax0right = axis[0].twinx()

ax0right.set_yticks([offs[0]])
ax0right.yaxis.set_major_locator(mticker.FixedLocator([offs[0]]))
ax0right.yaxis.set_major_formatter(mticker.FixedFormatter([r'$\Phi_0$']))
ax0right.set_ylim(axis[0].get_ylim()[0],axis[0].get_ylim()[-1])



axis[1].plot(xs * 1e6 ,numpy.zeros(xs.size),"--",dashes=(5,20),color='grey',linewidth=0.8)
#axis[1].plot(xs,dfevol1)
axis[1].plot(xs * 1e6,numpy.array(dfevol2) * 1e7)

axis[1].set_ylim(-1,1)

ax1right = axis[1].twinx()

ax1right.set_yticks([0.])
ax1right.yaxis.set_major_locator(mticker.FixedLocator([0.]))
ax1right.yaxis.set_major_formatter(mticker.FixedFormatter([r'']))
ax1right.set_ylim(axis[1].get_ylim()[0],axis[1].get_ylim()[-1])


axis[2].plot(xs * 1e6,ax(wdwtrans.freqs) * 1e-6,"--",dashes=(5,20),color='grey',linewidth=0.8)
#axis[2].plot(xs,sndderivevol1)
axis[2].plot(xs * 1e6,numpy.array(sndderivevol2) * 1e-6)

axis[2].set_ylim(20,45)

ax2right = axis[2].twinx()

ax2right.set_yticks([ax(freq[0]) * 1e-6])
ax2right.yaxis.set_major_locator(mticker.FixedLocator([ax(freq[0]) * 1e-6]))
ax2right.yaxis.set_major_formatter(mticker.FixedFormatter([r'$a$']))
ax2right.set_ylim(axis[2].get_ylim()[0],axis[2].get_ylim()[-1])

plt.savefig('png/transport_HOA2_objectives.png')

