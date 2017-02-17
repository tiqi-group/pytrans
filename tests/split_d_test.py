#!/usr/bin/env python3
# 
# Test out approach of generating many (alpha, beta) pairs along with
# their voltages, then characterising each set's separation for a pair
# of ions, and then producing the reverse function that gives a set of
# voltages for a given separation.

import sys
sys.path.append("../")
from pytrans import *
import splitting as sp

def split_d_test():
    """ This code has been incorporated into sp.split_waveforms_reparam() """
    # d stands for ion-ion distance
    # Generate trap polynomial approximations
    split_centre = -422.5*um
    electrode_subset = [3,4,5,6,7,18,19,20,21,22] # left splitting waveform
    polyfit_range = 200*um # spatial extent over which to fit polys to the solver
    polys = sp.generate_interp_polys(trap_mom.transport_axis,
                                     trap_mom.potentials[:, electrode_subset],
                                     split_centre, polyfit_range)

    # Generate range of alphas and positions
    n_alphas = 60
    alpha_range = np.hstack([np.linspace(1e7, 1e6, n_alphas//3),
                             np.linspace(0.9e6, -1.9e6, n_alphas//3),
                             np.linspace(-2e6, -1e7, n_alphas//3)])
    true_alphas = np.empty(n_alphas)
    true_betas = np.empty(n_alphas)
    split_elec_voltages = np.empty((len(electrode_subset), n_alphas))
    slope_offset = 0

    for k, alpha in enumerate(alpha_range):
        split_elec_voltages[:,[k]], true_alphas[k], true_betas[k] = sp.solve_poly_ab(
            polys, alpha, slope_offset)

    def plot_sep_fn(alpha, beta):
        """ Manually look at separation landscape for a given
        alpha/beta; zero crossing gives d value for Eq. (7) for
        Home/Steane paper (2003) """

        d = np.linspace(0, 500*um, 100000)
        offs = sp.ion_sep(d, alpha, beta)
        plt.plot(d/um, offs)
        plt.grid(True)
        plt.xlabel('Dist (um)')
        plt.ylabel("Equation offset (correct separation when it's equal to 0")
        plt.show()

    separations = sp.get_sep(true_alphas, true_betas)
    freqs = sp.com_w(separations, true_alphas, true_betas, mass_Ca)/2/np.pi

    # Desired separation profile: sin**2
    v_spl = sp.v_splrep(split_elec_voltages, separations)
    tau = np.linspace(0, 1, 200)
    sep_desired = separations[0] + (separations[-1]-separations[0])*tau**2*(np.sin(np.pi/2*tau))**2
    # sep_desired = separations[0] + (separations[-1]-separations[0])*tau
    v_interp = sp.v_splev(v_spl, sep_desired)

    st()
    spline_plots = False
    if spline_plots:
        if False:
            # Plot voltages as fn of distance
            # plt.plot(alpha_range, split_elec_voltages.T)
            
            plt.figure()
            sep_desired = np.linspace(separations[0], separations[-1], 5000)
            plt.plot(sep_desired, sp.v_splev(sep_v, sep_desired).T)
            plt.grid(True)
            plt.xlabel('Separation')
            plt.ylabel('Voltages')

        if True:
            # Plot freqs as fn of distance
            plt.figure()
            plt.plot(separations, freqs, ':x')
            plt.grid(True)
            plt.xlabel('Separation')
            plt.ylabel('Freqs (MHz)')
        
        plt.show()

    plot_seps = False
    if plot_seps:
        plt.plot(separations/um, true_alphas)
        plt.xlabel('Separations (um)')
        plt.ylabel('Alphas')
        plt.grid(True)

    plt.show()

if __name__ == "__main__":
    split_d_test()
