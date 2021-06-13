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

import matplotlib.pylab as plt
import matplotlib.ticker as mticker
from matplotlib import rc
import matplotlib

## Expremient settings
timesteps = 400
pos = numpy.array([-0.0004,0.0004])
freq = numpy.array([1.6e6,1.6e6])
offs = numpy.array([1.,1.])
linspace_fn=numpy.linspace


weights=[20000,18000,500,20000,500,100,0,200,0,0] # default weights used in the SolvePotentials2 routine (betas)
dfull=3.8e-3
dpart=7e-3

global_settings['USESOLVE2'] = False

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
mu1 = wf_wdwtrans.raw_samples()

# solver2:


weights=[20000,50000,5000,2000,500,100,0,20,0,0] # default weights used in the SolvePotentials2 routine (betas)
dfull=50e-3
dpart=100e-3

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

# offset relax



weights=[   0,1800000, 5000,2000, 500,100,0,200,0,1e4] # offset relaxation
dfull=5e-3
dpart=10e-3

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
muos = wf_wdwtrans.raw_samples()

# plotting
ax= lambda nu: 2* (2 * np.pi * nu)**2 * (mass_Ca* atomic_mass_unit)/(2 * electron_charge)

sol_mus = mu1
ys1 = []

dim =30
xs = range(timesteps)

for i in range(dim):
    y = numpy.zeros(timesteps)

    for x in xs:
        y[x] = sol_mus[i,x]
    ys1.append(y)





# plotting
ax= lambda nu: 2* (2 * np.pi * nu)**2 * (mass_Ca* atomic_mass_unit)/(2 * electron_charge)


sol_mus = mu2
ys2 = []

dim =30
xs = range(timesteps)

for i in range(dim):
    y = numpy.zeros(timesteps)

    for x in xs:
        y[x] = sol_mus[i,x]
    ys2.append(y)


# plotting
ax= lambda nu: 2* (2 * np.pi * nu)**2 * (mass_Ca* atomic_mass_unit)/(2 * electron_charge)


sol_mus = muos
ysos = []

dim =30
xs = range(timesteps)

for i in range(dim):
    y = numpy.zeros(timesteps)

    for x in xs:
        y[x] = sol_mus[i,x]
    ysos.append(y)


def plotevol(xs,ys,name):

    f = plt.figure(figsize=(10,5))
    f.subplots_adjust(left=0.07)
    f.subplots_adjust(right=0.85)
    axi = f.add_subplot(111)


    j = 7
    i = j
    k = 0

    while k < 7:

        axi.plot(xs,ys[k],'--',label='L'+str(j-k))
        k = k+1

    axi.plot(xs,ys[j],label='DCCa7')

    while i < 14:
        i = i+1
        axi.plot(xs,ys[i],'--', label='R'+str(i-j))

    axi.set_ylim(0,trap.Vmax+0.2)

    rightx = axi.twinx()
    rightx.set_yticks([trap.Vdefault,trap.Vmax])
    rightx.yaxis.set_major_locator(mticker.FixedLocator([0.,trap.Vmax]))
    rightx.yaxis.set_major_formatter(mticker.FixedFormatter([r'$V_{default}$',r'$V_{max}$']))
    rightx.set_ylim(axi.get_ylim()[0],axi.get_ylim()[-1])

    axi.plot(numpy.array(xs),numpy.ones(timesteps) * trap.Vmax,"--",dashes=(5,20),color='grey',linewidth=0.8)

    axi.set_xlabel('timesteps')
    axi.set_ylabel('voltage applied to electrode [V]')

    axi.legend(bbox_to_anchor=(1.0,0.9))
    plt.savefig('png/' + name + '.png')
    #plt.show()


plotevol(xs,ys1,'transport_ETH3d_solver1_evol')
plotevol(xs,ys2,'transport_ETH3d_solver2_evol')
plotevol(xs,ysos,'transport_ETH3d_solver_rel_offset_evol')

# prepare data of solver1
mus1 = [] #contains mu vector for every time step

timesteps =  400 # how many timesteps
#dim = 30 # How many electrodes
dim = 30
xs = range(timesteps)

for x in xs:
    mul = numpy.zeros(dim)

    mul = mu1[:,x]
    mus1.append(mul)

xs1 = wdwtrans.positions

pot0= lambda x,mul: numpy.dot(mul.T , trap.Func( x,0) )
pot1= lambda x,mul: numpy.dot(mul.T , trap.Func( x,1) )
pot2= lambda x,mul: numpy.dot(mul.T , trap.Func( x,2) )

offsetevol1 = []
dfevol1 = []
sndderivevol1 =[]

for x,mu in zip(xs1,mus1):
    offsetevol1.append(numpy.asarray(pot0(x,mu)).reshape(-1))
    dfevol1.append(numpy.asarray(pot1(x,mu)).reshape(-1))
    sndderivevol1.append(numpy.asarray(pot2(x,mu)).reshape(-1))



### prepare data of solver2
mus2 = [] #contains mu vector for every time step

timesteps =  400 # how many timesteps
#dim = 30 # How many electrodes
dim = 30
xs = range(timesteps)

for x in xs:
    mul = numpy.zeros(dim)

    mul = mu2[:,x]
    mus2.append(mul)

xs2 = wdwtrans.positions

pot0= lambda x,mul: numpy.dot(mul.T , trap.Func( x,0) )
pot1= lambda x,mul: numpy.dot(mul.T , trap.Func( x,1) )
pot2= lambda x,mul: numpy.dot(mul.T , trap.Func( x,2) )

offsetevol2 = []
dfevol2 = []
sndderivevol2 =[]

for x,mu in zip(xs2,mus2):
    offsetevol2.append(numpy.asarray(pot0(x,mu)).reshape(-1))
    dfevol2.append(numpy.asarray(pot1(x,mu)).reshape(-1))
    sndderivevol2.append(numpy.asarray(pot2(x,mu)).reshape(-1))


### prepare data of relexted offset
musos = [] #contains mu vector for every time step

timesteps =  400 # how many timesteps
#dim = 30 # How many electrodes
dim = 30
xs = range(timesteps)

for x in xs:
    mul = numpy.zeros(dim)

    mul = muos[:,x]
    musos.append(mul)

xsos = wdwtrans.positions

pot0= lambda x,mul: numpy.dot(mul.T , trap.Func( x,0) )
pot1= lambda x,mul: numpy.dot(mul.T , trap.Func( x,1) )
pot2= lambda x,mul: numpy.dot(mul.T , trap.Func( x,2) )

offsetevolos = []
dfevolos = []
sndderivevolos =[]

for x,mu in zip(xsos,musos):
    offsetevolos.append(numpy.asarray(pot0(x,mu)).reshape(-1))
    dfevolos.append(numpy.asarray(pot1(x,mu)).reshape(-1))
    sndderivevolos.append(numpy.asarray(pot2(x,mu)).reshape(-1))



import matplotlib.pylab as plt

#### TODO reset limits and where axis should be equal!


### Plot Solver1 vs Solver2

f1, axis1 = plt.subplots(3,1,sharex=True,figsize=(6.5,11))


f1.subplots_adjust(left=0.15)
axis1[0].xaxis.set_major_locator(plt.MaxNLocator(8))
axis1[0].xaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.1f$\mu m$'))
axis1[0].yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$V$'))
axis1[1].yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$\frac{V}{m}$'))
axis1[2].yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.0f $10^{6}\frac{V}{m^2}$'))

axis1[0].plot(xsos*1e6,wdwtrans.offsets,"--",dashes=(5,20),color='grey',linewidth=0.8)
#axis[0].plot(xs,offsetevol1)
axis1[0].plot(xsos*1e6,offsetevol2,label='FO method')
axis1[0].plot(xsos*1e6,offsetevol1,label='grid method')

axis1[0].set_ylim(0.7,1.1)

ax10right = axis1[0].twinx()

ax10right.set_yticks([offs[0]])
ax10right.yaxis.set_major_locator(mticker.FixedLocator([offs[0]]))
ax10right.yaxis.set_major_formatter(mticker.FixedFormatter([r'$\Phi_0$']))
ax10right.set_ylim(axis1[0].get_ylim()[0],axis1[0].get_ylim()[-1])



axis1[1].plot(xsos * 1e6 ,numpy.zeros(xsos.size),"--",dashes=(5,20),color='grey',linewidth=0.8)
#axis[1].plot(xs,dfevol1)
axis1[1].plot(xsos * 1e6,numpy.array(dfevol2) )
axis1[1].plot(xsos * 1e6,numpy.array(dfevol1) )

axis1[1].set_ylim(-10,10)

ax11right = axis1[1].twinx()

ax11right.set_yticks([0.])
ax11right.yaxis.set_major_locator(mticker.FixedLocator([0.]))
ax11right.yaxis.set_major_formatter(mticker.FixedFormatter([r'']))
ax11right.set_ylim(axis1[1].get_ylim()[0],axis1[1].get_ylim()[-1])


axis1[2].plot(xsos * 1e6,ax(wdwtrans.freqs) * 1e-6,"--",dashes=(5,20),color='grey',linewidth=0.8)
#axis[2].plot(xs,sndderivevol1)
axis1[2].plot(xsos * 1e6,numpy.array(sndderivevol2) * 1e-6)
axis1[2].plot(xsos * 1e6,numpy.array(sndderivevol1) * 1e-6)

#axis1[2].set_ylim(20,45)

ax12right = axis1[2].twinx()

ax12right.set_yticks([ax(freq[0]) * 1e-6])
ax12right.yaxis.set_major_locator(mticker.FixedLocator([ax(freq[0]) * 1e-6]))
ax12right.yaxis.set_major_formatter(mticker.FixedFormatter([r'$a$']))
ax12right.set_ylim(axis1[2].get_ylim()[0],axis1[2].get_ylim()[-1])

axis1[0].legend()
plt.savefig('png/' + 'transport_ETH3d_solver1v2_objectives' + '.png')

### PLot Solver2 vs Solver2 with offset relax

f2, axis2 = plt.subplots(3,1,sharex=True,figsize=(6.5,11))
f2.subplots_adjust(left=0.15)

axis2[0].xaxis.set_major_locator(plt.MaxNLocator(8))
axis2[0].xaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.1f$\mu m$'))
axis2[0].yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$V$'))
axis2[1].yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$\frac{V}{m}$'))
axis2[2].yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.0f $10^{6}\frac{V}{m^2}$'))

axis2[0].plot(xs2*1e6,wdwtrans.offsets,"--",dashes=(5,20),color='grey',linewidth=0.8)
axis2[0].plot(xs2*1e6,offsetevol2, label='FO method')
axis2[0].plot(xs2*1e6,offsetevolos,'r', label='FO with relaxed offset')

#axis2[0].set_ylim(0.7,1.1)

ax20right = axis2[0].twinx()

ax20right.set_yticks([offs[0]])
ax20right.yaxis.set_major_locator(mticker.FixedLocator([offs[0]]))
ax20right.yaxis.set_major_formatter(mticker.FixedFormatter([r'$\Phi_0$']))
ax20right.set_ylim(axis2[0].get_ylim()[0],axis2[0].get_ylim()[-1])



axis2[1].plot(xs2 * 1e6 ,numpy.zeros(xs2.size),"--",dashes=(5,20),color='grey',linewidth=0.8)
axis2[1].plot(xs2 * 1e6,numpy.array(dfevol2), label='FO method')
axis2[1].plot(xs2 * 1e6,numpy.array(dfevolos),'r' , label='FO with relaxed offset')

axis2[1].set_ylim(-10,10)

ax21right = axis2[1].twinx()

ax21right.set_yticks([0.])
ax21right.yaxis.set_major_locator(mticker.FixedLocator([0.]))
ax21right.yaxis.set_major_formatter(mticker.FixedFormatter([r'']))
ax21right.set_ylim(axis2[1].get_ylim()[0],axis2[1].get_ylim()[-1])


axis2[2].plot(xs2 * 1e6,ax(wdwtrans.freqs) * 1e-6,"--",dashes=(5,20),color='grey',linewidth=0.8)
axis2[2].plot(xs2 * 1e6,numpy.array(sndderivevol2) * 1e-6)
axis2[2].plot(xs2 * 1e6,numpy.array(sndderivevolos) * 1e-6,'r')

axis2[2].set_ylim(axis1[2].get_ylim()[0],axis1[2].get_ylim()[-1])

ax22right = axis2[2].twinx()

ax22right.set_yticks([ax(freq[0]) * 1e-6])
ax22right.yaxis.set_major_locator(mticker.FixedLocator([ax(freq[0]) * 1e-6]))
ax22right.yaxis.set_major_formatter(mticker.FixedFormatter([r'$a$']))
ax22right.set_ylim(axis2[2].get_ylim()[0],axis2[2].get_ylim()[-1])

axis2[1].legend()
plt.savefig('png/' + 'transport_ETH3d_solver2v_rel_off_objectives' + '.png')
