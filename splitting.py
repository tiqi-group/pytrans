#!/usr/bin/env python3
# Splitting library, building on pytrans' existing functionality
# (NOT FINISHED!)

import sys
sys.path.append("./")
from pytrans import *
import scipy.optimize as sopt

########## OLD CODE (used to investigate Home/Steane paper's tradeoffs) ###########

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






######## NEW CODE (developed mainly in testing/splitting_constraints_test.py) #########

import scipy.interpolate as interp

def calc_beta(sep, alp):
    # sep: separation from ion to ion
    # alp: prefactor in quadratic
    return  2*np.abs(alp)/(sep)**2
    
def interpolate_moments(z_shifted, moments, roi_idxes, order=4):
        return np.polyfit(z_shifted[roi_idxes], moments[roi_idxes,:], order)

def generate_interp_polys(z_axis, moments, centre, roi_dist):
    # z_axis: axis corresponding to moments
    # moments: M x N matrix, M = number of z points, N = number of trap electrodes (may be reduced)
    # centre: z axis centre around which to fit polynomials (same units as z_axis)
    # roi_dist: total z range over which to fit polynomials (same units as z_axis)
    #      (not needed if solving for the whole trap)
    # returns the polynomial fit around the centre, assuming the z axis is zero at the centre
    roi_idxes = np.arange(np.argmax(z_axis > centre - roi_dist/2),
                          np.argmax(z_axis > centre + roi_dist/2))
    polys = interpolate_moments(z_axis-centre, moments, roi_idxes)
    return polys

def solve_scaled_constraints(moments, desired_pot, offset, scale_weight):
    # All array variables should be defined over the same set of x
    # coordinates (which are arbitrary)
    #
    # moments: electrode moments, M x N array, M = x coordinates,
    # N = number of (relevant)electrodes
    #
    # desired_pot: desired potential well kernel, which will be
    # scaled as large as electrode limits can support
    #
    # offset: potential offset required after scaling
    # scale_weight: how heavily to weight the scaling vs the fitting
    # Returns a column vector (2d) of the optimal electrode values
    states = []
    uopt = cvy.Variable(moments.shape[1], 1)
    vscale = cvy.Variable()
    cost = cvy.sum_squares(moments*uopt
                           - desired_pot*vscale - offset) # * vscale previously
    constr = [-max_elec_voltages[0] <= uopt, uopt <= max_elec_voltages[0]] # should make non-hardcoded
    # constr += [0.1 <= vscale]
    states.append(cvy.Problem(cvy.Minimize(cost-scale_weight*vscale), constr))
    prob = sum(states)        
    prob.solve(solver='ECOS', verbose=False)
    return uopt.value, vscale.value

def solve_poly_ab(poly_moments, alpha=0, slope_offset=None, dc_offset=None,
                  print_voltages=False, enforce_z_symmetry=False,
                  verbose_solver=False):
    # slope_offset: extra slope (electric field) to apply along z
    # direction, in V/m (can read it right off the potential
    # plots)
    #
    # dc_offset: extra potential to apply, in V (can read it right
    # off the plots)
    #
    # alpha: quadratic curvature (note: if the magnitude is too
    # large, the solver may fail)
    num_elec = poly_moments.shape[1]
    uopt = cvy.Variable(num_elec,1)
    # Quadratic and quartic terms in poly approximations
    alph_c = poly_moments[2,:]
    beta_c = poly_moments[0,:]
    gamm_c = poly_moments[3,:]
    dc_c = poly_moments[4,:]
    # for some reason normalisation is needed
    alph_norm = np.min(np.abs(alph_c))
    beta_norm = np.min(np.abs(beta_c))
    gamm_norm = np.min(np.abs(gamm_c))
    dc_norm = np.min(np.abs(dc_c))
    alph_co = alph_c/alph_norm
    beta_co = beta_c/beta_norm
    gamm_co = gamm_c/gamm_norm
    dc_co = dc_c/dc_norm
    # Avoid going over-voltage
    constr = [-max_elec_voltages[0] <= uopt, uopt <= max_elec_voltages[0]]

    # electrode constraint pairs assume electrode moments are
    # adjacent, in order of increasing z and with the bottom row
    # following the top row. Additionally electrodes must be
    # symmetric around splitting zone.

    # Ensure x-y symmetric pairs of electrodes agree 
    for k in range(num_elec//2):
        constr.append(uopt[k] == uopt[k+num_elec//2])
    # Ensure z-symmetric (around splitting zone) pairs of electrodes agree
    if enforce_z_symmetry:
        for m in range(num_elec//4):
            constr.append(uopt[m] == uopt[-m-1])
    obj = cvy.Maximize(cvy.sum_entries(beta_co*uopt))
    obj -= 100*cvy.Minimize(cvy.sum_squares(alph_co*uopt - alpha/alph_norm))
    # constr.append(cvy.sum_entries(alph_co*uopt)==alpha/alph_norm) # quadratic term
    if dc_offset:
        constr.append(cvy.sum_entries(dc_co*uopt)==dc_offset/dc_norm) # linear term ~= 0
    if slope_offset:
        # obj -= cvy.Minimize(cvy.sum_squares(gamm_co*uopt - slope_offset/gamm_norm))
        # polyder_moments = np.vstack((np.polyder(k) for k in poly_moments.T)).T
        obj -= 10*cvy.Minimize(cvy.sum_squares(gamm_co*uopt - slope_offset/gamm_norm))
        
    # obj = cvy.Maximize(cvy.sum_entries(beta_co*uopt))+cvy.Maximize(-cvy.sum_entries(alph_co*uopt))
    prob = cvy.Problem(obj, constr)
    prob.solve(solver=global_solver, verbose=verbose_solver)
    ans = uopt.value
    if print_voltages:
        print("Voltages: ", uopt.value[:num_elec//2])
    assert ans is not None, "cvxpy did not find a solution."
    return ans, np.sum(alph_co*ans)*alph_norm, np.sum(beta_co*ans)*beta_norm

if __name__ == "__main__":
    #reproduce_fig2_home_steane()
    # look_at_wells_manually()
    wp = WaveformSet(waveform_file="waveform_files/load_split_2Be2Ca_2016_07_06_v04.dwc.json").find_waveform("apart").samples[:,[174]]
    find_coulomb_wells(wp, -422.5*um, 100*um)
