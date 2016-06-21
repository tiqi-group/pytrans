#!/usr/bin/env python3
# Splitting library, building on pytrans' existing functionality
# (NOT FINISHED!)

import sys
sys.path.append("./")
from pytrans import *
import scipy.optimize as sopt

def scan_alpha_beta():
    # Look at standard
    spatial_range = 20*um
    z = np.linspace(-spatial_range/2,spatial_range/2,1000)
    a_b_traj = ((1,1e-6),(0,1),(-1,100))
    a_sc = 1e8
    b_sc = 1e18
    ramp_pts = 3

    for (a_start,b_start), (a_end, b_end) in zip(a_b_traj[:-1], a_b_traj[1:]):
        a_ramp = np.linspace(a_start, a_end, ramp_pts)*a_sc
        b_ramp = np.linspace(b_start, b_end, ramp_pts)*b_sc

        for a, b in zip(a_ramp, b_ramp):
            plt.plot(z, a*z**2+b*z**4)
            rough_d = np.sqrt(np.abs(a)/2/b)

            #st()
            
            print("rough d = ", rough_d, ", d = ", sopt.root(lambda k: dist(a, b, k), rough_d).x )

    plt.show()


def ion_distance_ab(alpha, beta, d):
    if d<0: # to constrain the solver
        return abs(d)*1e5+1e2
    return beta*d**5+2*alpha*d**3-electron_charge/2/np.pi/epsilon_0
    
def ion_distance_as(d, s, alpha):
    d2s = d/2/s
    return d2s**5 + np.sign(alpha)*d2s**3 - electron_charge/4/np.pi/epsilon_0/np.abs(alpha)/(2*s)**3

def freq_to_alpha(freq):
    return freq**2*atomic_mass_unit*40/2/electron_charge

def reproduce_fig2_home_steane():
    alpha_x = np.linspace(-1.4e8, 1.4e8, 100)
    beta = 2.7e18

    w1_y = np.zeros_like(alpha_x)
    d_y = np.zeros_like(alpha_x)

    s_x = np.sqrt(np.abs(alpha_x)/2/beta)
    for k, (s, alpha) in enumerate(zip(s_x,alpha_x)):
        d = np.abs(sopt.root(lambda m: ion_distance_as(m, s, alpha), s).x)
        d_y[k] = d

    w1_y = np.sqrt(2*alpha_x + 3*beta*d_y**2*electron_charge/atomic_mass_unit/40)

    st()

def look_at_wells_manually():
    alpha = 1.4e8
    beta = 2.65e18

    def trap_freqs(z_axis, pot):
        # pot += np.random.random(z_axis.size)*1e-10 # to avoid argrelmin getting stuck
        # Re-implementing stuff in WavPotential class
        pot_resolution=z_axis[1]-z_axis[0]
        potg2 = np.gradient(np.gradient(pot))#self.pot_resolution**2
        min_indices = ssig.argrelextrema(pot, np.less_equal, order=20)
        offsets = potg2[min_indices]
        grads = potg2[min_indices]/pot_resolution**2
        trap_freqs = np.sqrt(electron_charge*grads / (40*atomic_mass_unit))/2/np.pi
        trap_locs = z_axis[min_indices]

        return {'min_indices':min_indices, 'offsets':offsets, 'freqs':trap_freqs, 'locs':trap_locs}

    z_axis = np.linspace(-20,20,1000)*um
    single_pot = alpha*z_axis**2
    double_pot = -alpha*z_axis**2+beta*z_axis**4

    st()
    print(trap_freqs(z_axis, single_pot))
    print(trap_freqs(z_axis, double_pot))

if __name__ == "__main__":
    #reproduce_fig2_home_steane()
    look_at_wells_manually()
