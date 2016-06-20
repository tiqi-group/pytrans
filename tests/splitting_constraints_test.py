#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

import scipy.interpolate as interp
import matplotlib.animation as animation

def splitting_constraints_test():
    # Simulate 5 electrodes, each with a very simple Gaussian envelope
    z_axis = np.linspace(-100,100,1000)*um
    e0_moment = np.exp(-(z_axis - (-60))**2 / 30**2)
    e1_moment = np.exp(-(z_axis - (-3))**2 / 30**2)
    e2_moment = np.exp(-(z_axis - (0))**2 / 30**2)
    e1_moment = np.exp(-(z_axis - (-3))**2 / 30**2)
    e2_moment = np.exp(-(z_axis - (0))**2 / 30**2)

def trap_freqs(z_axis, pot):
    if len(pot.shape) > 1: # flatten into a 1d array
        pot = np.ravel(pot)
    # pot += np.random.random(z_axis.size)*1e-10 # to avoid argrelmin getting stuck
    # Re-implementing stuff in WavPotential class
    pot_resolution=z_axis[1]-z_axis[0]
    st()
    potg2 = np.gradient(np.gradient(pot))#self.pot_resolution**2
    # min_indices = ssig.argrelmin(pot, order=20)
    min_indices = ssig.argrelextrema(pot, np.less_equal, order=20)
    offsets = potg2[min_indices]
    grads = potg2[min_indices]/pot_resolution**2
    trap_freqs = np.sqrt(electron_charge*grads / (40*atomic_mass_unit))/2/np.pi
    trap_locs = z_axis[min_indices]

    return {'min_indices':min_indices, 'offsets':offsets, 'freqs':trap_freqs, 'locs':trap_locs}    

if __name__ == "__main__":
    splitting_constraints_test()

    # z_axis = np.linspace(-100,100,1000)*um
    # e0_moment = np.exp(-(z_axis - (-60*um))**2 / (45*um)**2)
    # e1_moment = np.exp(-(z_axis - (-30*um))**2 / (45*um)**2)
    # e2_moment = np.exp(-(z_axis - (0*um))**2 / (45*um)**2)
    # e3_moment = np.exp(-(z_axis - (30*um))**2 / (45*um)**2)
    # e4_moment = np.exp(-(z_axis - (60*um))**2 / (45*um)**2)        
    # moments = np.column_stack([e0_moment, e1_moment, e2_moment, e3_moment, e4_moment])
    z_axis = trap_mom.transport_axis
    moments = trap_mom.potentials

    def calc_betas(seps, alph):
        beta = np.zeros_like(alph)
        for k, (sep, alp) in enumerate(zip(seps, alph)):
            beta[k] = np.abs(alp)/2/sep
        return beta            
    
    # separations = np.array([0, 10, 20, 30, 40, 50])*um
    # alphas = np.array([1.4e8, 1.4e8, 1.4e8, -1.4e8, -1.4e8, -1.4e8])    
    # betas = np.zeros_like(alphas)
    # for k, (sep, alph) in enumerate(zip(separations[1:], alphas[1:])):
    #     betas[k+1] = np.abs(alph)/2/sep**2        
    # print(betas)
    # voffsets = np.array([0, 0, 0, 0, 0, 0])
    # roi_dists = np.array([20, 20, 20, 20, 20, 20])*um
    # scale_weights = np.array([1, 1, 1, 1, 1, 1])

    separations = np.array([0])*um
    #alphas = np.array([1.4e4])
    alphas = np.array([1.4e8])
    # betas = calc_betas(separations, alphas)
    betas = np.array([0])
    voffsets = np.array([0])
    roi_dists = np.array([100])*um
    scale_weights = np.array([1])

    # Prepare fine moments
    res_increase = 10
    z_axis_interp = np.linspace(z_axis[0], z_axis[-1], len(z_axis)*res_increase)
    moments_interp = np.zeros((moments.shape[0]*res_increase, moments.shape[1]))    
    for k, mom in enumerate(moments.T):
        moments_interp[:,k] = interp.splev(z_axis_interp, interp.splrep(z_axis, mom), der=0)    
    potential_quad = z_axis_interp**2
    potential_quart = z_axis_interp**4

    for a, b, offs, roi_dist, sw in zip(alphas, betas, voffsets, roi_dists, scale_weights):    
        states = []
        uopt = cvy.Variable(moments.shape[1], 1)
        vscale = cvy.Variable()
        
        roi_idxes = np.arange(np.argmax(z_axis_interp > -roi_dist/2), np.argmax(z_axis_interp > roi_dist/2))
        # Generate interpolated dataset for the moments
        moments_int = np.zeros_like(moments_interp[roi_idxes,:])
        cost = cvy.sum_squares(moments_interp[roi_idxes,:]*uopt
                               - (a*potential_quad[roi_idxes]
                                  + b*potential_quart[roi_idxes])*vscale - offs) # * vscale previously
        constr = [-8.5 <= uopt, uopt <= 8.5]
        # constr += [0.1 <= vscale]
        states.append(cvy.Problem(cvy.Minimize(cost-sw*vscale), constr))
        prob = sum(states)
        
        prob.solve(solver='ECOS', verbose=False)

        ans = uopt.value
        for m, u in zip(moments_interp.T, np.ravel(ans)):
            col = m.T
            plt.plot(z_axis_interp, col*u,':')

        # pot = (potential_quad*a+potential_quart*b)*vscale.value + offs
        pot = moments_interp*ans
            
        plt.plot(z_axis_interp, pot, 'g') # vscale.value
        plt.plot(z_axis_interp, moments_interp*ans, 'r')

        plt.gca().set_xlim([-800*um, 800*um])
        plt.gca().set_ylim([-20, 20])
        plt.title("Alpha = "+str(a)+", beta = "+str(b)+", offset = "+str(offs))

        print("voltages: ", ans)
        print("vscale: ", vscale.value)
        print("trap freq(s): ", trap_freqs(z_axis_interp, pot)['freqs']/MHz, " MHz")
        plt.show()
