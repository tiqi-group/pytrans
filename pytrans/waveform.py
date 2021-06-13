#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np
import cvxpy as cx

from .potential_well import PotentialWell

import logging
logger = logging.getLogger(__name__)


class Waveform:
    """
    doc
    """

    def __init__(self) -> None:
        self.harmonic_wells = []
        self.samples = 0

    def add_harmonic_well(self, freq, pos, offs, energy_threshold):
        well = PotentialWell(freq, pos, offs, energy_threshold)
        self.harmonic_wells.append(well)
        self.samples = max(self.samples, well.samples)
        return well

    def static_solver(self, trap, r0, default_V=None,
                      solver="MOSEK", verbose=True,
                      **kwargs):
        """Static solver

        Args:
        moments

        Returns:
        uopt
        """
        x = trap.transport_axis
        num_electrodes = trap.num_electrodes
        default_V = trap.default_V if default_V is None else default_V
        # uopt = cx.Variable(shape=(self.samples, num_electrodes), name="uopt_static")
        uopt = cx.Variable(num_electrodes, name="uopt_static")

        # static voltage cost
        cost = cx.sum_squares(cx.multiply(np.sqrt(r0), uopt - default_V))
        # potential cost
        for hw in self.harmonic_wells:
            cost += cx.sum_squares(
                cx.multiply(
                    hw.gaussian_weight(x),
                    uopt @ trap.moments[:, hw.roi(x)] - hw.potential(x)))
        # setup constrains
        constr = [trap.min_V <= uopt, uopt <= trap.max_V]
        # for j in range(1, 5):
        #     constr += [uopt[j] == 0]
        for j in range(num_electrodes // 2):
            constr += [uopt[j] == uopt[10 + j]]
        # TODO: symmetry constraint
        # constr += [uopt[:15, :] == uopt[15:, :]]
        prob = cx.Problem(cx.Minimize(cost), constr)
        logger.info("Setting up static problem")
        # if settings['solver_printnd_problem']:
        #     print("Potentials: ", potentials, "\nROIs: ", roi, "\nWeights: ", weights)
        # print("End prob: ", prob)
        prob.solve(solver=solver, verbose=verbose, **kwargs)
        # uoptv = uopt.value
        return uopt

    def transport_solver(self, trap, r0,
                         r1, r2,
                         solver="MOSEK", verbose=True,
                         **kwargs):
        pass
