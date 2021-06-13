#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Module docstring

"""

import numpy as np
import matplotlib.pyplot as plt

import cvxpy as cx

from .utils.timer import timer
import logging

logger = logging.getLogger(__name__)


@timer
def static_solver(moments, potential, roi, gaussian_weight, r0_weight, default_V=5,
                  min_V=-10, max_V=10, solver="MOSEK", verbose=True, **kwargs):
    """Static solver

    Args:
      moments

    Returns:
      uoptv
    """
    num_electrodes = moments.shape[0]
    uopt = cx.Variable(num_electrodes, name="uopt_static")
    constr = [min_V <= uopt, uopt <= max_V]
    # TODO: symmetry constraint
    # constr += [uopt[:15, :] == uopt[15:, :]]
    cost = cx.sum_squares(cx.multiply(np.sqrt(r0_weight), uopt - default_V))
    cost += cx.sum_squares(
        cx.multiply(gaussian_weight, uopt @ moments[:, roi] - potential)
    )
    prob = cx.Problem(cx.Minimize(cost), constr)
    logger.info("Setting up static problem")
    # if settings['solver_printnd_problem']:
    #     print("Potentials: ", potentials, "\nROIs: ", roi, "\nWeights: ", weights)
    # print("End prob: ", prob)
    prob.solve(solver=solver, verbose=verbose, **kwargs)
    # uoptv = uopt.value
    return uopt


@timer
def transport_solver(moments,
                     potentials, rois, gaussian_weights,
                     r0_weights, default_V=5,
                     min_V=-10, max_V=10,
                     solver="MOSEK",
                     verbose=True,
                     **kwargs
                     ):
    """

    Args:
      moments

    Returns:
      uoptv
    """
    len_t = potentials.shape[0]
    len_x, num_electrodes = moments.shape
    uopt = cx.Variable((len_t, num_electrodes), name="uopt_transport")
    # TODO: symmetry
    cost = cx.sum_squares(cx.multiply(np.sqrt(r0_weights), uopt - default_V))
  return cost

# def _static_solver(
#     min_elec_voltages, max_elec_voltages, sw_r0_u_ss_m, potentials, roi, weights
# ):
#     uopt_e = cx.Variable((wdp.num_electrodes, 1))
#     constr_e = [min_elec_voltages <= uopt_e, uopt_e <= max_elec_voltages]
#     constr_e += [uopt_e[:15, :] == uopt_e[15:, :]]
#     # penalise deviations from default voltage
#     cost_e = cx.sum_squares(
#         sw["r0_u_weights"] * (uopt_e - sw_r0_u_ss_m) * np.sqrt(sw["r0"])
#     )
#     cost_e += cx.sum_squares(
#         cx.multiply(weights, trap_mom.potentials[roi] * uopt_e - potentials)
#     )
#     # was ist das?
#     # cost_e += cx.sum_squares(trap_mom.potentials[
#     #     wdp.roi_idx[-1]] * uopt_e[:,-1] - wdp.potentials[-1])
#     prob = cx.Problem(cx.Minimize(cost_e), constr_e)
#     if settings["solver_print_end_problem"]:
#         print("Potentials: ", potentials, "\nROIs: ", roi, "\nWeights: ", weights)
#         # print("End prob: ", prob)
#     prob.solve(solver=settings["solver"], verbose=settings["solver_verbose"], **kwargs)
#     uopt_ev = uopt_e.value
#     return uopt_ev
