#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

import scipy.interpolate as interp
import matplotlib.animation as animation

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

def solve_poly_ab(poly_moments, alpha=0, slope_offset=None, dc_offset=None, print_voltages=True):
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
    for m in range(num_elec//4):
        constr.append(uopt[m] == uopt[-m-1])
    constr.append(cvy.sum_entries(alph_co*uopt)==alpha/alph_norm) # quadratic term
    if slope_offset:
        constr.append(cvy.sum_entries(gamm_co*uopt)==slope_offset/gamm_norm) # linear term ~= 0
    if dc_offset:
        constr.append(cvy.sum_entries(dc_co*uopt)==dc_offset/dc_norm) # linear term ~= 0
    obj = cvy.Maximize(cvy.sum_entries(beta_co*uopt))        
    # obj = cvy.Maximize(cvy.sum_entries(beta_co*uopt))+cvy.Maximize(-cvy.sum_entries(alph_co*uopt))
    prob = cvy.Problem(obj, constr)
    prob.solve(solver='CVXOPT', verbose=True)
    ans = uopt.value
    if print_voltages:
        print("Voltages: ", uopt.value[:num_elec//2])
    return ans, np.sum(alph_co*ans)*alph_norm, np.sum(beta_co*ans)*beta_norm

def splitting_constraints_test():
    z_axis = trap_mom.transport_axis

    electrode_subset = [3,4,5,6,7,18,19,20,21,22] # left splitting group
    electrode_subset = [4,5,6,19,20,21] # only inner 3 narrow electrodes
    moments = trap_mom.potentials[:, electrode_subset]
    
    # separations = np.array([0, 10, 20, 30, 40, 50])*um
    # alphas = np.array([1.4e8, 1.4e8, 1.4e8, -1.4e8, -1.4e8, -1.4e8])    
    # betas = np.zeros_like(alphas)
    # for k, (sep, alph) in enumerate(zip(separations[1:], alphas[1:])):
    #     betas[k+1] = np.abs(alph)/2/sep**2        
    # print(betas)
    # voffsets = np.array([0, 0, 0, 0, 0, 0])
    # roi_dists = np.array([20, 20, 20, 20, 20, 20])*um
    # scale_weights = np.array([1, 1, 1, 1, 1, 1])

    # alpha = 1.4e8, beta = 0 gives a deep well

    z_centre = -422.6*um # centre of the central splitting electrode moment
    separations = np.array([100])*um # defined as distance from well to well
    #alphas = np.array([1.4e4])
    # alphas = np.array([-1e10])
    alphas = np.array([2e8])
    #betas = calc_beta(separations, alphas)
    betas = np.array([1e17])
    voffsets = np.array([0])
    roi_dists = 2*separations
    scale_weights = np.array([1])    

    # Prepare fine moments
    res_increase = 10
    z_axis_interp = np.linspace(z_axis[0], z_axis[-1], len(z_axis)*res_increase)
    z_shifted_interp = z_axis_interp-z_centre 
    moments_interp = np.zeros((moments.shape[0]*res_increase, moments.shape[1]))    
    for k, mom in enumerate(moments.T):
        # using z_axis_interp/z_axis instead of z_shifted_interp/z_shifted makes no difference
        moments_interp[:,k] = interp.splev(z_axis_interp, interp.splrep(z_axis, mom), der=0)    
    potential_quad = z_shifted_interp**2
    potential_quart = z_shifted_interp**4

    for a, b, offs, roi_dist, sw in zip(alphas, betas, voffsets, roi_dists, scale_weights):        
        roi_idxes = np.arange(np.argmax(z_shifted_interp > -roi_dist/2), np.argmax(z_shifted_interp > roi_dist/2))
        # Generate interpolated dataset for the moments
        # polys = interpolate_moments(z_shifted_interp, moments_interp, roi_idxes)
        polys = generate_interp_polys(z_axis, trap_mom.potentials[:, electrode_subset],
                                      z_centre, roi_dist)
        test_polys = False
        if test_polys:
            for poly in polys.T:
                plt.plot(z_shifted_interp, np.polyval(poly, z_shifted_interp))
            plt.plot(z_shifted_interp, moments_interp)
            plt.gca().set_ylim([-0.02,0.2])
            plt.show()

        polys_2der = np.zeros((polys.shape[0]-2, polys.shape[1]))
        for k, poly in enumerate(polys.T):
            polys_2der[:,k] = np.polyder(np.polyder(poly))
        test_polys_deriv = False
        if test_polys_deriv:
            z_interp_step = z_shifted_interp[1]-z_shifted_interp[0]
            for poly in polys_2der.T:
                plt.plot(z_shifted_interp, np.polyval(poly, z_shifted_interp))
            for moment in moments_interp.T:
                plt.plot(z_shifted_interp, np.gradient(np.gradient(moment))/z_interp_step**2) # 2 shorter
            plt.gca().set_ylim([-1e7,5e6])
            plt.show()
            
        ans_grid, scale = solve_scaled_constraints(
            moments_interp[roi_idxes,:],
            a*potential_quad[roi_idxes] + b*potential_quart[roi_idxes],
            offs, sw)
        # Waveform conditions
        # pre-splitting, still one well at 1.35 MHz
        ans_poly_before_split, alpha, beta = solve_poly_ab(polys, 1.5e7, slope_offset=None, dc_offset=None)
        # alpha = 0, beta maximised, wells equalised by an extra slope
        ans_poly_mid_split, alpha, beta = solve_poly_ab(polys, 0, slope_offset=0.16, dc_offset=None)
        # slightly after splitting
        ans_poly_after_split, alpha, beta = solve_poly_ab(polys, -3e6, slope_offset=None, dc_offset=None)
        # post splitting, 
        ans_poly_after_split2, alpha, beta = solve_poly_ab(polys, -1e7, slope_offset=None, dc_offset=None)
        ans_poly_after_split3, alpha, beta = solve_poly_ab(polys, -5e7, slope_offset=None, dc_offset=None)
        ans = ans_poly_after_split3
        
        for m, u in zip(moments_interp.T, np.ravel(ans)):
            col = m.T
            plt.plot(z_axis_interp/um, col*u,':')

        pot_desired = (potential_quad*a+potential_quart*b)*scale + offs
        pot = moments_interp*ans

        plt.plot(z_axis_interp/um, pot_desired, 'g')
        plt.plot(z_axis_interp/um, pot, 'r')

        plt.gca().set_xlim([-800+z_centre/um, 800+z_centre/um])
        plt.gca().set_ylim([-20, 20])
        plt.title("Alpha eff = "+str(a*scale)+", beta = "+str(b*scale)+", offset = "+str(offs))

        all_voltages = np.zeros(30)
        all_voltages[electrode_subset] = ans
        print("voltages: ", all_voltages[:15])
        print("vscale: ", scale)

        tfd = find_wells(moments*ans, np.array(z_axis[1]-z_axis[0]), 40, mode='precise', freq_threshold=5*kHz)
        poss = []
        freqs = []
        for loc, freq in zip(tfd['locs'], tfd['freqs']):
            if z_centre - roi_dists < loc < z_centre + roi_dists: # twice the width of solver ROI
                poss.append(loc/um)
                freqs.append(freq/MHz)
        print("trap pos(s): ", poss, " um")
        print("trap freq(s): ", freqs, " MHz")
        plt.show()

if __name__ == "__main__":
    splitting_constraints_test()
