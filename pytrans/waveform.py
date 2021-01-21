#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Module docstring

"""

import numpy as np
import matplotlib.pyplot as plt
from pytrans import *
from pytrans.units import *
from pytrans.global_settings import *



class WavDesired:
    """ Specifications describing potential wells to solve for"""

    def __init__(self,
                 potentials,  # list of arrays; each array is a potential for a timestep; volts
                 weights,  # list of arrays; each weights is applied to the potentials to adjust how the solver weighs each error
                 roi_idx,  # Element indices for global trap axis position array; dims must match potentials, weights
                 Ts=200 * ns,  # slowdown of 0 -> 10 ns/step, slowdown of 19 (typical) -> (10*(19+1)) = 200 ns/step
                 mass=mass_Ca,
                 num_electrodes=30,
                 desc=None,
                 solver_weights=None,
                 force_static_ends=True):  # force solver result for 1st + last timesteps to be equal to the static case (exclude all effects like slew rate etc)
        self.desc = desc
        self.potentials = potentials
        self.weights = weights
        self.roi_idx = roi_idx
        self.Ts = Ts
        self.mass = mass
        self.num_electrodes = num_electrodes
        if desc:
            self.desc = desc
        else:
            self.desc = "No description specified"
        self.solver_weights = {
            # Cost function parameters
            'r0': 1e-15,  # punishes deviations from r0_u_ss. Can be used to set default voltages for less relevant electrodes.
            'r1': 1e-5,  # punishes the first time derivative of u, thus reducing the slew rate
            'r2': 0,  # punishes the second time derivative of u, thus further enforcing smoothness

            # default voltage for the electrodes. any deviations from
            # this will be punished, weighted by r0 and r0_u_weights
            'r0_u_ss': np.ones(num_electrodes) * default_elec_voltage,  # default voltages for the electrodes
            'r0_u_weights': np.ones(num_electrodes)  # use this to put different weights on outer electrodes
        }
        if solver_weights:
            # non-default solver parameters
            self.solver_weights.update(solver_weights)
        self.force_static_ends = force_static_ends

    def plot(self, trap_axis, ax=None):
        """ ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.plot(trap_axis[self.roi_idx] / um, self.potentials)
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        return ax


class WavDesiredWells(WavDesired):
    def __init__(self,
                 # array or list of arrays/lists (can be single-element) of each position as a fn of timestep
                 positions,
                 freqs,  # array, same dimensions as positions
                 offsets,  # array, same dimensions as positions
                 desired_potential_params={},
                 Ts=200 * ns,  # slowdown of 0 -> 10 ns/step, slowdown of 19 (typical) -> (10*(19+1)) = 200 ns/step
                 mass=mass_Ca,
                 num_electrodes=30,
                 desc=None,
                 solver_weights=None,
                 force_static_ends=True,
                 anharmonic_terms=[],
                 start_pot=[None],
                 end_pot=[None],
                 # solver2_weights=[20000,18000,500,20000,500,100,0,200,0,0], # default weights used in the SolvePotentials2 routine (betas)
                 # solver2_weights=[   0,18000, 500,20000, 500,100,0,200,0,1e6], # offset relaxation
                 solver2_weights=[50000, 18000, 1000, 20000, 1000, 100, 0, 20, 0, 0],  # HOA2 weights
                 # d_full=30e-3, #Mainz Voltag restriction off
                 # d_part=40e-3, #Mainz voltage restriction off
                 # d_full=10e-2,
                 # d_part=20e-2,
                 d_full=3.8e-3,
                 d_part=7e-3,
                 # d_full=5e-3,
                 # d_part=9e-3,
                 trap_m=None
                 ):

        self.trap = trap_m  # NOTE only used in the solver2 for now

        potentials, weights, roi_idx = self.desiredPotentials(positions, freqs, offsets,
                                                              mass, des_pot_parm=desired_potential_params,
                                                              anharmonic_terms=anharmonic_terms)

        # save the original data in WavDesired, so they can be accessed in Waveform.solve_potentials2
        # NOTE Multiple wells don't work with the new one for now
        self.positions = positions[0]
        self.freqs = freqs[0]
        self.offsets = offsets[0]
        self.start_Potential = start_pot[0]
        self.end_Potential = end_pot[0]
        self.solver2_weights = solver2_weights

        # if trap_m is None:
        #     self.trap = trap_mom
        # else:
        #     self.trap = trap_m

        # define the allowed voltages
        self.d_full = d_full
        self.d_part = d_part
        #              |-d_full-|
        #              |---d_part---|
        # _             .        .    ________
        # \            .        .   /
        #  \           .        .  /
        #   \          .        . /
        #    \___________________/

        super().__init__(potentials, weights, roi_idx, Ts, mass, num_electrodes,
                         desc, solver_weights, force_static_ends)

    def desiredPotentials(self, pos, freq, off, mass, des_pot_parm={}, anharmonic_terms=[]):
        # Rough threshold width for 1 SD in solver potential
        pot_params = dict(global_des_pot_settings)
        pot_params.update(des_pot_parm)
        energy_threshold = pot_params['energy_threshold']

        assert type(pos) is type(freq) is type(off), "Input types inconsistent"
        if type(pos) is list or tuple:
            # Construct 2D matrices from lists: columns of each are
            # the timesteps, rows of each are the discrete wells
            pos = np.vstack(pos).T
            freq = np.vstack(freq).T
            off = np.vstack(off).T

        # Estimate range of ROI indices to use
        a = (2 * np.pi * freq[pot_params['roi_timestep']][pot_params['roi_well']])**2 * (mass * atomic_mass_unit) / (2 * electron_charge)
        v_desired = a * (self.trap.transport_axis - pos[0][0])**2
        width_1sd = (v_desired < energy_threshold).sum()

        # distance 1 std dev from centre
        dist_1sd = width_1sd * (self.trap.transport_axis[1] - self.trap.transport_axis[0]) / 2
        width_roi = 4 * width_1sd  # extend ROI to 4x distance of 1 standard deviation from the central point
        if (width_roi % 2 == 0):
            width_roi += 1  # make sure it's an odd number

        dims = list(pos.shape)
        dims[1] *= width_roi
        pots = np.empty(dims)
        wghts = np.empty(dims)
        rois = np.empty(dims, dtype='int')

        for k, (po, fr, of) in enumerate(zip(pos, freq, off)):  # iterate over timesteps
            assert len(po) is not 0, "Desired wells supplied in incorrect format: must be list of lists or 2D array"

            for m, (po_l, fr_l, of_l) in enumerate(zip(po, fr, of)):  # iterate over discrete wells
                a = (2 * np.pi * fr_l)**2 * (mass * atomic_mass_unit) / (2 * electron_charge)
                x = self.trap.transport_axis - po_l
                v_desired = a * x**2 + of_l
                if anharmonic_terms:
                    v_desired += anharmonic_terms[0] * x**3
                central_idx = np.argmin(v_desired)  # could equally do argmin(abs(x axis - x centre))
                idces = np.arange(central_idx - width_roi // 2, central_idx + width_roi // 2 + 1, dtype=int)

                rois[k, m * width_roi:(m + 1) * width_roi] = idces
                # roi_lim_list.append(idces[[0,-1]])
                pots[k, m * width_roi:(m + 1) * width_roi] = v_desired[idces]

                wght_x = self.trap.transport_axis[idces] - po_l
                wght_gauss = np.exp(-wght_x**2 / (2 * dist_1sd**2))  # Gaussian centred on well centre
                wghts[k, m * width_roi:(m + 1) * width_roi] = wght_gauss

        return pots, wghts, rois

    def plot(self, idx=0, ax=None):
        """ ax: Matplotlib axes """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        ax.plot(self.trap.transport_axis[self.roi_idx[idx]] / um, self.potentials[idx])
        ax.set_xlabel('trap location (um)')
        ax.set_ylabel('potential (V)')
        return ax


class Waveform:
    """Waveform storage class. Convert an input list into a numpy array
    and store various details about it, or solve given a set of
    constraints and the global trap potentials.
    """

    def __init__(self, *args, **kwargs):
        if len(args) is 4:
            # Directly create Waveform from [desc, uid, samples, generated]
            self.desc = args[0]
            self.uid = args[1]
            self.generated = args[2]
            self.samples = np.array(args[3])
        elif isinstance(args[0],  WavDesired):  # check if a child of WavDesired
            # Create waveform based on WavDesired by setting up and solving an optimal control problem
            self.wdp = args[0]
            # Some informations on the last run of the Dual Program solver
            # Details SolverStats of cvxpy
            self.SolverStats = None
            self.SizeMetrics = None
            # get a log of of applied Voltage restrictions
            self.Vresmaxlog = []
            self.Vresminlog = []
            # solving
            raw_samples = self.solve_potentials(self.wdp, **kwargs)  # ordered by electrode
            num_elec, num_timesteps = raw_samples.shape
            self.samples = np.zeros((num_elec + 2, num_timesteps))  # Add two DEATH monitor channels
            self.samples[:, :] = raw_samples[list(abs(k) for k in self.wdp.trap.dac_channel_transform), :]  # Transform as required

            self.desc = self.wdp.desc
            self.set_new_uid()
            self.generated = ""

        else:
            assert False, "Need some arguments in __init__."

    def __repr__(self):
        return 'Wfm: "{d}" ({s} long)'.format(d=self.desc, s=self.samples.shape[1])

    def raw_samples(self):
        return self.samples[self.wdp.trap.physical_electrode_transform, :]

    def solve_potentials2(self, wdp, trap_m=None, **kwargs):
        """ a solve implementation that optimises controlling the potential characteristics instead of a grid"""
        settings = dict(global_settings)
        settings.update(kwargs)

        if wdp.trap is None:
            if trap_m is None:
                trap = trap_mom
            else:
                trap = trap_m
        else:
            if trap_m is None:
                trap = wdp.trap
            else:
                trap = trap_m

        print("starting Solver2 using trap: " + str(trap))
        t1 = timeit.default_timer()

        Vdef = trap.Vdefault

        def basic_cost(mu, beta, freq, pos, offset, trap, d_full, d_part, mass=1):
            """ Basic const function for one timestep"""

            # frequency to curvature
            a = (2 * np.pi * freq)**2 * (mass * atomic_mass_unit) / (2 * electron_charge)

            # function values needed
            delta = (1 / a)
            Fx = trap.Func((pos), 0)
            Fdx = trap.Func((pos - delta), 0)
            Fxd = trap.Func((pos + delta), 0)
            DFx = trap.Func((pos), 1)
            D2Fx = trap.Func((pos), 2)
            D2Fdx = trap.Func((pos - delta), 2)
            D2Fxd = trap.Func((pos + delta), 2)

            # Function value
            if (offset > 0.001 or offset < 0.001):
                cost = cvy.sum_squares((mu.T * Fx - offset) / offset)
            else:
                cost = cvy.sum_squares(mu.T * Fx - offset)

            cost *= beta[0]

            # D2F
            if (a > 1):
                c1 = cvy.sum_squares(mu.T * (D2Fx / abs(2 * a)) - 1)
                # D2F at deltas
                c2 = cvy.sum_squares(mu.T * (D2Fdx /
                                             abs(2 * a)) - 1)
                c3 = cvy.sum_squares(mu.T * (D2Fxd / abs(2 * a)) - 1)
            else:  # freq to close to 0
                c1 = cvy.sum_squares(mu.T * D2Fx - (2 * a))
                # D2F at deltas
                c2 = cvy.sum_squares(mu.T * D2Fdx - (2 * a))
                c3 = cvy.sum_squares(mu.T * D2Fxd - (2 * a))

            cost += beta[1] * c1 + beta[2] * (c2 + c3)

            # DF == 0
            cost += beta[3] * cvy.sum_squares(mu.T * DFx)
            # F symmetric
            cost += beta[4] * cvy.sum_squares(mu.T * Fdx - mu.T * Fxd)

            # regulize usage of mu
            cost += beta[5] * cvy.norm(mu - Vdef, 2)
            cost += beta[6] * cvy.norm(mu - Vdef, 1)

            # constrain the use of electrodes faraway from the Region of interest
            con = []

            # returns a double between [0,1] that scales the Voltage constrained depending on the RIO
            def get_VCon(xe, xdes):
                c = abs(xe - xdes)
                c -= 0.5 * d_full
                c /= 0.5 * (d_full - d_part)
                c = max([c, 0])
                c = min([c, 1])
                return c

            Vmaxres = np.zeros(mu.shape[0])
            Vminres = np.zeros(mu.shape[0])

            for i, Vmax, Vmin, xmid in zip(range(mu.shape[0]), trap.Vmaxs, trap.Vmins, trap.x_mids):

                v_con_min = Vdef
                v_con_max = Vdef
                # allow multiple locations for a single voltage (e.g. HOA2)
                r = 0
                if isinstance(xmid, list):
                    for x in xmid:
                        rn = get_VCon(x, pos)
                        if rn > r:
                            r = rn

                else:
                    # assuming only one float or double
                    r = get_VCon(xmid, pos)
                v_con_max = Vdef + (r * (Vmax - Vdef))
                v_con_min = Vdef - (r * (Vdef - Vmin))
                Vmaxres[i] = v_con_max
                Vminres[i] = v_con_min
                con.append(mu[i] <= v_con_max)
                con.append(mu[i] >= v_con_min)

            self.Vresmaxlog.append(Vmaxres)
            self.Vresminlog.append(Vminres)

            # make sure no unsupported voltages are applied
            for i in range(mu.shape[0]):
                con.append(mu[i] <= trap.Vmaxs[i])
                con.append(mu[i] >= trap.Vmins[i])

            return (cost, con)

        def offset_change_cost(mu0, mu1, pos0, pos1, trap):
            Fx0 = trap.Func((pos0), 0)
            Fx1 = trap.Func((pos1), 0)

            return cvy.square(mu0.T * Fx0 - mu1.T * Fx1)

        if isinstance(wdp.positions, (int, float)):
            N = 1
        else:
            N = len(wdp.positions)
        mu = cvy.Variable((trap.numberofelectrodes, N))
        beta = cvy.Parameter((10, 1), nonneg=True)
        cost = 0
        constraints = []

        # Global constraints
        assert (N < 2) or (N > 3), "Cannot have this number of timesteps, due to finite-diff approximations"
        if N == 1:
            static_ends = False  # static by default for 1 timestep
        else:
            static_ends = wdp.force_static_ends

        # NOTE does wdp really need num_electrodes? shouldn't it rather get the trap?
        # NOTE shouldn't wdp have a link to the trap?

        weights = wdp.solver2_weights
        weightlength = beta.shape[0]
        if weights is None:
            warnings.warn("No solver_weights were set for the desiered Waveform! Using unit vector")
            weights = np.ones(beta.shape)
        elif len(weights) is not weightlength:
            warnings.warn("the given solver_weights have wrong dimensions! Using unit vector")
            weights = np.ones(beta.shape)
            beta.value = weights
        else:
            beta.value = weights

        # individual steps
        if N == 1:
            # the static case (the positions,freqs and offsets aren't lists)
            c, con = basic_cost(mu[:, 0], beta, wdp.freqs, wdp.positions, wdp.offsets, trap, wdp.d_full, wdp.d_part, mass=wdp.mass)  # HeinekaS
            # c, con = basic_cost(mu,beta,wdp.freqs,wdp.positions,wdp.offsets,trap,wdp.d_full,wdp.d_part,mass = wdp.mass) # roswald
            cost += c
            constraints.extend(con)
        else:
            for i, pos, freq, offset in zip(range(N), wdp.positions, wdp.freqs, wdp.offsets):
                # caclucalet basic_costs
                c, con = basic_cost(mu[:, i], beta, freq, pos, offset, trap, wdp.d_full, wdp.d_part, mass=wdp.mass)  # HeinekaS
                # c, con = basic_cost(mu,beta,freq,pos,offset,trap,wdp.d_full,wdp.d_part,mass = wdp.mass) # roswald
                cost += c
                constraints.extend(con)

            # penalise changes between steps in first and second derivative
            cost += beta[7] * cvy.sum_squares(mu[:, :-1] - mu[:, 1:])
            cost += beta[8] * 1 / 4 * cvy.sum_squares(mu[:, :-2] - mu[:, 2:])

            # constrain maximum change in mu(t)
            constraints += [trap.max_slew_rate * wdp.Ts >= cvy.abs(mu[:, 1:] - mu[:, :-1])]

            for i in range(N - 1):
                cost += beta[9] * offset_change_cost(mu[:, i], mu[:, i + 1], wdp.positions[i], wdp.positions[i + 1], trap)

        # Make ends match static case or given voltages
        if static_ends:
            def get_static(f, p, o):
                stat_mu = cvy.Variable((trap.numberofelectrodes, 1))
                c, con = basic_cost(stat_mu, beta, f, p, o, trap, wdp.d_full, wdp.d_part, mass=wdp.mass)
                skiplist = []
                for i in range(trap.numberofelectrodes):
                    if i not in skiplist:
                        j = trap.symmetry[i]
                        con += [stat_mu[i, :] == stat_mu[j, :]]
                        skiplist.append(j)
                cvy.Problem(cvy.Minimize(c), con).solve(solver=settings['solver'], verbose=settings['solver_verbose'], **kwargs)
                return stat_mu.value

            if (getattr(wdp, "start_Potential", None) is None):
                wdp.start_Potential = get_static(wdp.freqs[0], wdp.positions[0], wdp.offsets[0])
            if (getattr(wdp, "end_Potential", None) is None):
                wdp.end_Potential = get_static(wdp.freqs[-1], wdp.positions[-1], wdp.offsets[-1])

            # constrain the ends
            constraints += [mu[:, 0] == wdp.start_Potential]
            constraints += [mu[:, -1] == wdp.end_Potential]

        # Enforce absolute symmetry
        skiplist = []
        for i in range(trap.numberofelectrodes):
            if i not in skiplist:
                j = trap.symmetry[i]
                constraints += [mu[i, :] == mu[j, :]]
                skiplist.append(j)

        #constraints = []
        t2 = timeit.default_timer()
        print("setup time: ", t2 - t1)
        # solve Optimisation Problem
        prob = cvy.Problem(cvy.Minimize(cost), constraints)
        t3 = timeit.default_timer()
        print("setup time solver: ", t3 - t2)
        prob.solve(solver=settings['solver'], verbose=settings['solver_verbose'], **kwargs)
        t4 = timeit.default_timer()
        print("solve time: ", t4 - t3)
        print("total time: ", t4 - t1)

        # saving information for debugging

        self.cost = cost
        self.constraints = constraints
        self.problem = prob
        self.SolverStats = prob.solver_stats
        self.SizeMetrics = prob.size_metrics
        return mu.value

    # This is the legacy version, that optimizes the well on a grid, given a configured well
    def solve_potentials1(self, wdp, **kwargs):
        """ Convert a desired set of potentials and ROIs into waveform samples
        wdp: waveform desired potential"""
        # TODO: make this more flexible, i.e. arbitrary-size voltages
        max_slew_rate = 5 / us  # (units of volts / s, quarter of DEATH AD8021 op-amps)

        settings = dict(global_settings)
        settings.update(kwargs)

        trap_mom = wdp.trap
        # Cost function parameters
        sw = wdp.solver_weights
        if settings['solver_print_weights']:
            print("Solver weights: ", sw)

        N = len(wdp.potentials)  # timesteps

        # Set up and solve optimisation problem
        uopt = cvy.Variable((wdp.num_electrodes, N))
        states = []

        # Global constraints
        assert (N < 2) or (N > 3), "Cannot have this number of timesteps, due to finite-diff approximations"
        if N == 1:
            static_ends = False  # static by default for 1 timestep
        else:
            static_ends = wdp.force_static_ends

        # Penalise deviations from default voltage
        sw_r0_u_ss_m = np.tile(sw['r0_u_ss'], (N, 1)).T  # matrixized
        # cost = sw['r0']*cvy.sum_squares(sw['r0_u_weights'] * (uopt - sw_r0_u_ss_m))
        cost = cvy.sum_squares(sw['r0_u_weights'] * (uopt - sw_r0_u_ss_m) * np.sqrt(sw['r0']))

        # Absolute voltage constraints
        min_elec_voltages_m = np.tile(min_elec_voltages, (N, 1)).T
        max_elec_voltages_m = np.tile(max_elec_voltages, (N, 1)).T

        constr = []
        if static_ends:
            constr += [min_elec_voltages_m[:, 1:-1] <= uopt[:, 1:-1],
                       uopt[:, 1:-1] <= max_elec_voltages_m[:, 1:-1]]

            # Absolute symmetry constraints
            constr += [uopt[:15, 1:-1] == uopt[15:, 1:-1]]
        else:
            constr += [min_elec_voltages_m <= uopt, uopt <= max_elec_voltages_m]

            # Absolute symmetry constraints
            constr += [uopt[:15, :] == uopt[15:, :]]

        # Constrain the end voltages explicitly to match static case
        # (i.e. solve separate problems first, then constrain main one)
        def get_boundary_voltages(min_elec_voltages, max_elec_voltages, sw_r0_u_ss_m, potentials, roi, weights):
            uopt_e = cvy.Variable((wdp.num_electrodes, 1))
            constr_e = [min_elec_voltages <= uopt_e, uopt_e <= max_elec_voltages]
            constr_e += [uopt_e[:15, :] == uopt_e[15:, :]]
            # penalise deviations from default voltage
            cost_e = cvy.sum_squares(sw['r0_u_weights'] * (uopt_e - sw_r0_u_ss_m) * np.sqrt(sw['r0']))
            cost_e += cvy.sum_squares(cvy.multiply(weights, trap_mom.potentials[roi] * uopt_e - potentials))
            # cost_e += cvy.sum_squares(trap_mom.potentials[
            #     wdp.roi_idx[-1]] * uopt_e[:,-1] - wdp.potentials[-1])
            prob = cvy.Problem(cvy.Minimize(cost_e), constr_e)
            if settings['solver_print_end_problem']:
                print("Potentials: ", potentials, "\nROIs: ", roi, "\nWeights: ", weights)
                # print("End prob: ", prob)
            prob.solve(solver=settings['solver'], verbose=settings['solver_verbose'], **kwargs)
            uopt_ev = uopt_e.value
            return uopt_ev

        def old_solver():
            uopt_e = cvy.Variable((wdp.num_electrodes, 2))
            # min_elec_voltages_e = np.tile(min_elec_voltages, (2,1)).T
            # max_elec_voltages_e = np.tile(max_elec_voltages, (2,1)).T
            min_elec_voltages_e = min_elec_voltages_m[:, [0, -1]]
            max_elec_voltages_e = max_elec_voltages_m[:, [0, -1]]

            constr_e = [min_elec_voltages_e <= uopt_e, uopt_e <= max_elec_voltages_e]  # absolute voltage
            constr_e += [uopt_e[:15, :] == uopt_e[15:, :]]
            sw_r0_u_ss_m_e = sw_r0_u_ss_m[:, [0, -1]]
            # penalise deviations from default voltage
            cost_e = cvy.sum_squares(sw['r0_u_weights'] * (uopt_e - sw_r0_u_ss_m_e) * np.sqrt(sw['r0']))
            cost_e += cvy.sum_squares(trap_mom.potentials[
                wdp.roi_idx[0]] * uopt_e[:, 0] - wdp.potentials[0])
            cost_e += cvy.sum_squares(trap_mom.potentials[
                wdp.roi_idx[-1]] * uopt_e[:, -1] - wdp.potentials[-1])
            prob = cvy.Problem(cvy.Minimize(cost_e), constr_e)
            prob.solve(solver=settings['solver'], verbose=settings['solver_verbose'], **kwargs)
            uopt_ev = uopt_e.value
            return uopt_ev

        if static_ends:
            uopt_start = get_boundary_voltages(min_elec_voltages_m[:, [0]], max_elec_voltages_m[:, [0]], sw_r0_u_ss_m[:, [0]],
                                               wdp.potentials[0], wdp.roi_idx[0], wdp.weights[0])
            uopt_end = get_boundary_voltages(min_elec_voltages_m[:, [-1]], max_elec_voltages_m[:, [-1]], sw_r0_u_ss_m[:, [-1]],
                                             wdp.potentials[-1], wdp.roi_idx[-1], wdp.weights[-1])
            uopt_ev = np.hstack([uopt_start, uopt_end])

            # Currently there is a bug: uopt_ev_old is not equal to uopt_ev at the moment. Not a huge problem for now.
            #uopt_ev_old = old_solver()

            # Add constraint
            constr += [uopt[:, [0, -1]] == uopt_ev]

        # Approximate costs on first and second derivative of u with finite differences
        # Here, we use 2nd order approximations. For a table with coefficients see
        # https://en.wikipedia.org/wiki/Finite_difference_coefficient

        if N > 3:
            # Middle: central finite-difference approx
            cost += cvy.sum_squares(np.sqrt(sw['r1']) * 0.5 * (uopt[:, 2:] - uopt[:, :-2]))  # deriv
            cost += cvy.sum_squares(np.sqrt(sw['r2']) * (uopt[:, 2:] - 2 * uopt[:, 1:-1] + uopt[:, :-2]))  # 2nd deriv

            # Start: use forward finite difference approximation of derivatives
            cost += cvy.sum_squares(np.sqrt(sw['r1']) * (-0.5 * uopt[:, 2] + 2 * uopt[:, 1] - 1.5 * uopt[:, 0]))
            cost += cvy.sum_squares(np.sqrt(sw['r2']) * (-uopt[:, 3] + 4 * uopt[:, 2] - 5 * uopt[:, 1] + 2 * uopt[:, 0]))

            # End: use backward finite difference approximation of derivatives
            cost += cvy.sum_squares(np.sqrt(sw['r1']) * (1.5 * uopt[:, -1] - 2 * uopt[:, -2] + 0.5 * uopt[:, -3]))
            cost += cvy.sum_squares(np.sqrt(sw['r2']) * (2 * uopt[:, -1] - 5 * uopt[:, -2] + 4 * uopt[:, -3] - uopt[:, -4]))

            # Slew rate penalty
            constr += [max_slew_rate * wdp.Ts >= cvy.abs(uopt[:, 1:] - uopt[:, :-1])]
            # cost += 0.1*cvy.sum_squares(cvy.abs(uopt[:,1:]-uopt[:,:-1]))

        if False:
            pot_weights = np.ones(len(wdp.potentials))
            weight_ends = False
            if weight_ends and not static_ends:
                # Weight the constraints more heavily at the ending timesteps
                if len(wdp.potentials) > 6:
                    pot_weights[[0, -1]] = 1e7
                    pot_weights[[1, -2]] = 1e5
                    pot_weights[[2, -3]] = 1e4
                    pot_weights[[3, -4]] = 1e3
                    pot_weights[[4, -5]] = 1e2
                    pot_weights[[5, -6]] = 10
        # for kk, (pot, roi, weight) in enumerate(zip(wdp.potentials, wdp.roi_idx, pot_weights)):
        # st()
        # roi_moments = # CONTINUE HERE
        # cvy.sum_squares(cvy.multiply(wdp.weights.T, roi_moments*uopt - wdp.potentials.T

        for kk, (pot, roi, weights) in enumerate(zip(wdp.potentials, wdp.roi_idx, wdp.weights)):
            # Cost term capturing how accurately we generate the desired potential well
            # (could also vectorise it like the above, but the ROI
            # indices tend to vary in length between timesteps)
            # cost += weight * cvy.sum_squares(trap_mom.potentials[roi, :]*uopt[:,kk] - pot)
            cost += cvy.sum_squares(cvy.multiply(weights, trap_mom.potentials[roi, :] * uopt[:, kk] - pot))

        states.append(cvy.Problem(cvy.Minimize(cost), constr))

        # ECOS is faster than CVXOPT, but can crash for larger problems
        prob = sum(states)
        print_debug("Whole prob: ", prob)
        try:
            prob.solve(solver=settings['solver'], verbose=settings['solver_verbose'], **kwargs)
        except cvy.error.SolverError:
            st()

        if False:
            # DEBUGGING ONLY, TRACKING DESIRED CONSTRAINTS
            plt.plot(trap_mom.transport_axis, trap_mom.potentials * uopt.value[:, 0], '--')
            plt.plot(trap_mom.transport_axis[wdp.roi_idx[0]], wdp.potentials[0])

        if settings['solver_check_end_constraints_met']:
            # DEBUGGING ONLY, ENSURING END CONSTRAINTS ARE MET
            if static_ends:
                print("Difference between expected and observed start/end constraints: {:.8f}".format(np.abs(uopt.value[:, [0, -1]] - uopt_ev).sum()))

        if prob.status == 'infeasible':
            warnings.warn("Waveform problem is infeasible. Try reducing assumed slowdown.")
            st()

        return uopt.value

    # an interface function that makes it easy to switch between solve_potentials versions
    def solve_potentials(self, wdp, **kwargs):
        if global_settings['USESOLVE2']:
            return self.solve_potentials2(wdp, **kwargs)
        else:
            return self.solve_potentials1(wdp, **kwargs)

    def set_new_uid(self):
        self.uid = np.random.randint(0, 2**31)

    def voltage_limits_exceeded(self):
        for column in self.samples.T:
            if np.any(column > max_death_voltages) or np.any(column < min_death_voltages):
                return True
        return False


class WaveformSet:
    """Waveform set handler, both for pre-generated JSON waveform files
    and waveform sets dynamically generated in Python"""

    def __init__(self, *args, **kwargs):
        """Can be used in several modes.

        The first is to generate a WaveformSet from an existing
        waveform file; the keyword argument waveform_file must have
        the filepath of the file to read.

        The second is to supply a list of existing Waveforms. These
        will be sorted by their names and checked for consistency.
        """

        if 'waveform_file' in kwargs.keys():
            # Create WaveformSet from existing JSON file
            with open(kwargs['waveform_file']) as fp:
                self.json_data = json.load(fp)
                waveform_num = len(self.json_data.keys())
                self.waveforms = []  # zero-indexed, unlike Matlab and Ionizer
                for k in range(1, waveform_num + 1):
                    jd = self.json_data['wav' + str(k)]
                    self.waveforms.append(Waveform(
                        jd['description'],
                        int(jd['uid'], 16),
                        jd['generated'],
                        jd['samples']
                    ))

        elif type(args[0][0]) is Waveform:
            # Use existing list of Waveforms (no transfer of ownership for now!)
            # for k, wf in enumerate(args[0]):
            #     assert wf.desc is 'wav'+str(k+1), "Waveforms are not ordered. TODO: auto-order them"
            self.waveforms = args[0]

        else:
            # Nothing could be understood
            assert False, "Couldn't parse input args"

    def __repr__(self):
        ret = ""
        for k in self.waveforms:
            ret += k.__repr__() + "\n"
        return ret

    def write(self, file_path, fix_voltage_limits=False):
        if os.path.isfile(file_path):
            warnings.warn("File " + file_path + " already exists. Overwriting...")
        with open(file_path, 'w') as fp:
            wfm_dict = {}
            total_samples_written = 0
            for k, wf in enumerate(self.waveforms):
                min_v = np.tile(min_death_voltages - max_overhead, (wf.samples.shape[1], 1)).T
                max_v = np.tile(max_death_voltages + max_overhead, (wf.samples.shape[1], 1)).T
                wfv_too_low = wf.samples < min_v
                wfv_too_high = wf.samples > max_v

                if fix_voltage_limits:
                    wf.samples[wfv_too_high] = max_v[wfv_too_high]
                    wf.samples[wfv_too_low] = max_v[wfv_too_low]

                fix_str = ""
                if fix_voltage_limits:
                    fix_str = " Truncating voltages to limit values specified in pytrans.py."

                if np.any(wfv_too_low):
                    warnings.warn("{k} DEATH voltages too low! May not load in Ionizer. {s}".format(k=wfv_too_low.sum(), s=fix_str))

                if np.any(wfv_too_high):
                    warnings.warn("{k} DEATH voltages too high! May not load in Ionizer. {s}".format(k=wfv_too_high.sum(), s=fix_str))

                total_samples_written += wf.samples.shape[1]
                if total_samples_written > max_death_samples:
                    warnings.warn('Too many DEATH samples desired; truncating waveform file at Waveform ' + str(k + 1))
                    break

                wfm_dict['wav' + str(k + 1)] = {
                    'description': wf.desc,
                    'uid': hex(wf.uid),  # cut off 0x to suit ionizer
                    'generated': wf.generated,
                    'samples': wf.samples.tolist()}
            json.dump(wfm_dict, fp, indent="", sort_keys=True)

    def get_waveform(self, num):
        """Return the waveform specified by a 1-indexed string or 0-indexed
        int. Accepts strings ('wav2') or ints (1).
        """
        if type(num) is str:
            idx = int(num[3:]) - 1
        elif type(num) is int:
            idx = num

        # assert idx >= 0, "Cannot access negative waveforms. Supply a 1-indexed string or 0-indexed int."
        return self.waveforms[idx]

    def find_waveform(self, name_str, get_index=False):
        """ Tries to find a waveform whose description partially or fully matches name_str """
        for k, w in enumerate(self.waveforms):
            if name_str in w.desc:
                if get_index:
                    return k
                return w
        warnings.warn("Could not find Waveform {w} in WaveformSet!".format(w=name_str))

    def find_waveforms(self, name_str, get_index=False):
        """ Tries to find a list of waveforms whose description partially or fully matches name_str """
        matching_wfms = []
        for k, w in enumerate(self.waveforms):
            if name_str in w.desc:
                if get_index:
                    matching_wfms.append(k)
                else:
                    matching_wfms.append(w)
        if not matching_wfms:
            warnings.warn("Could not find Waveform {w} in WaveformSet!".format(w=name_str))
        return matching_wfms
