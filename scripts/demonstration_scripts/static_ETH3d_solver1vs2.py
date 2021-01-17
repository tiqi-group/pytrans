#!/usr/bin/env python3
# static with ETH3d using solver1
# author: Sebastian Heinekamp

# imports
from scipy.interpolate import splrep, splev
import matplotlib
from matplotlib import rc
import matplotlib.ticker as mticker
from HOA2Trap import HOA2Trap as HOA2Moments
from ETH3dTrap import ETH3dTrap as ETHMoments
import loading_utils as lu
import transport_utils as tu
import copy as cp
from reorder import *
from pytrans import *
import numpy

import sys
sys.path.append("../")

# Loading conveyor stuff

# available Traps


# Expremient settings
pos = -0.000
freq = 2000000
offset = 1


weights = [20000, 18000, 2000, 20000, 500, 100, 0, 200, 0, 0]  # default weights used in the SolvePotentials2 routine (betas)
# solver2_weights=[   0,18000, 500,20000, 500,100,0,200,0,1e6], # offset relaxation
# solver2_weights=[50000,18000,1000,20000,1000,100,0, 20,0,0], #  HOA2 weights
dfull = 30e-3  # Mainz Voltag restriction off
dpart = 40e-3  # Mainz voltage restriction off
# d_full=10e-2,
# d_part=20e-2,
# dfull=3.8e-3
# dpart=7e-3
# dfull=5e-3
# dpart=9e-3

# print(global_settings['USESOLVE2'])
global_settings['USESOLVE2'] = False

trap = ETHMoments()
trap_mom.overwriteGlobalVariables()  # makes sure the global variables correspond to the trap variables - only needed if not executed in Moments constructor

wdw2 = WavDesiredWells([pos], [freq], [offset], trap_m=trap, solver2_weights=weights, d_full=dfull, d_part=dpart)
wf = Waveform(wdw2)

mu1 = wf.raw_samples()


# Solver2
# same experiment for solver 2
global_settings['USESOLVE2'] = True

trap = ETHMoments()
trap.overwriteGlobalVariables()  # makes sure the global variables correspond to the trap variables - only needed if not executed in Moments constructor

wdw2 = WavDesiredWells([pos], [freq], [offset], trap_m=trap, solver2_weights=weights, d_full=dfull, d_part=dpart)
wf = Waveform(wdw2)

mu2 = wf.raw_samples()


# print

# calculacte errors
a = 2 * (2 * np.pi * freq)**2 * (mass_Ca * atomic_mass_unit) / (2 * electron_charge)

# calculacte errors
print("The errors for solver1: ")
print("The offset error (normalized): ")
print((numpy.dot(mu1.T, trap.Func(pos, 0)) - offset) / offset)
print("The first derivative at the well bottom")
print(numpy.dot(mu1.T, trap.Func(pos, 1)))
print("The frequency error (normalized): ")
print((numpy.dot(mu1.T, trap.Func(pos, 2)) - a) / a)
print("The error for solver2:")
print("The offset error (normalized): ")
print((numpy.dot(mu2.T, trap.Func(pos, 0)) - offset) / offset)
print("The first derivative at the well bottom")
print(numpy.dot(mu2.T, trap.Func(pos, 1)))
print("The frequency error (normalized): ")
print((numpy.dot(mu2.T, trap.Func(pos, 2)) - a) / a)

# Plotting


#rc ('text', usetex=True)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

gridsize = (3, 1)
plt.figure(figsize=(6.3, 7.9))
potax = plt.subplot2grid(gridsize, (0, 0))
fieldax = plt.subplot2grid(gridsize, (1, 0), sharex=potax)
freqax = plt.subplot2grid(gridsize, (2, 0), sharex=potax)

potax.xaxis.set_major_locator(plt.MaxNLocator(7))
potax.xaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$\mu m$'))
potax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$V$'))
fieldax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$\frac{V}{m}$'))
freqax.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.0f$10^{6}\frac{V}{m^2}$'))


def fx(mu, x, deriv): return numpy.dot(mu.T, trap.Func(x, deriv))


xs = numpy.linspace(-0.2 * 10e-3, 0.2 * 10e-3, 400)
potxs = numpy.array(list(map(lambda x: fx(mu1, x, 0), xs)))
pot2xs = numpy.array(list(map(lambda x: fx(mu2, x, 0), xs)))
dpotxs = numpy.array(list(map(lambda x: fx(mu1, x, 1), xs)))
d2potxs = numpy.array(list(map(lambda x: fx(mu1, x, 2), xs)))
dpot2xs = numpy.array(list(map(lambda x: fx(mu2, x, 1), xs)))
d2pot2xs = numpy.array(list(map(lambda x: fx(mu2, x, 2), xs)))


i = 0

potax.plot(xs * 1e6, potxs, label='grid method')
potax.plot(xs * 1e6, pot2xs, label='FO method')

potaxright = potax.twinx()
potax.plot(xs * 1e6, numpy.array([offset] * 400), '--', dashes=(5, 20), color='grey', linewidth=0.8)
potax.axvline(x=pos, linestyle='--', dashes=(5, 20), color='grey', linewidth=0.8)
potaxright.set_yticks([offset])
# potaxright.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'$\Phi_0$'))
potaxright.yaxis.set_major_locator(mticker.FixedLocator([offset]))
potaxright.yaxis.set_major_formatter(mticker.FixedFormatter([r'$\Phi_0$']))
potaxright.set_ylim(potax.get_ylim()[0], potax.get_ylim()[-1])


# plt.ylim([-0.001,0.001])
fieldax.plot(xs * 1e6, dpotxs)
fieldax.plot(xs * 1e6, dpot2xs)

fieldaxright = fieldax.twinx()
fieldax.plot(xs * 1e6, numpy.array([0.0] * 400), '--', dashes=(5, 20), color='grey', linewidth=0.8)
fieldax.axvline(x=pos, linestyle='--', dashes=(5, 20), color='grey', linewidth=0.8)
fieldaxright.set_yticks([0.0])
# fieldaxright.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'$\Phi_0$'))
fieldaxright.yaxis.set_major_locator(mticker.FixedLocator([0.0]))
fieldaxright.yaxis.set_major_formatter(mticker.FixedFormatter([r'']))
fieldaxright.set_ylim(fieldax.get_ylim()[0], fieldax.get_ylim()[-1])

freqax.plot(xs * 1e6, d2potxs * 1e-6)
freqax.plot(xs * 1e6, d2pot2xs * 1e-6)

freqaxright = freqax.twinx()
freqax.plot(xs * 1e6, numpy.array([a * 1e-6] * 400), '--', dashes=(5, 20), color='grey', linewidth=0.8)
freqax.axvline(x=pos, linestyle='--', dashes=(5, 20), color='grey', linewidth=0.8)
freqaxright.set_yticks([a * 1e-6])
# freqaxright.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'$\Phi_0$'))
freqaxright.yaxis.set_major_locator(mticker.FixedLocator([a * 1e-6]))
freqaxright.yaxis.set_major_formatter(mticker.FixedFormatter([r'$a$']))
freqaxright.set_ylim(freqax.get_ylim()[0], freqax.get_ylim()[-1])

potax.legend()
plt.savefig('png/static_ETH3d_solver1vs2.png')
plt.show()
