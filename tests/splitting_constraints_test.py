#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *
from splitting import *

def splitting_constraints_test():
    z_axis = trap_mom.transport_axis

    electrode_subset = [3,4,5,6,7,18,19,20,21,22] # left splitting group
    # electrode_subset = [4,5,6,19,20,21] # only inner 3 narrow electrodes
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
        ans = ans_poly_before_split
        
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
        offsets = []
        for loc, freq, off in zip(tfd['locs'], tfd['freqs'], tfd['offsets']):
            if z_centre - roi_dists < loc < z_centre + roi_dists: # twice the width of solver ROI
                poss.append(loc/um)
                freqs.append(freq/MHz)
                offsets.append(off/meV)
        print("trap pos(s): ", poss, " um")
        print("trap freq(s): ", freqs, " MHz")
        print("trap offset(s): ", offsets, " meV")
        plt.show()

if __name__ == "__main__":
    splitting_constraints_test()
