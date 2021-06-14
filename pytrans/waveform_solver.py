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

from pytrans.conversion import curv_to_freq

import logging
logger = logging.getLogger(__name__)

h_weight = np.asarray([
    [0, 1e-3, 1e-3],
    [1e-3, 1, 1],
    [1e-3, 1, 1]
]).ravel()


class Solver:
    """
    doc
    """

    def __init__(self, trap, wells):
        assert len(set([w.samples for w in wells])) <= 1, "All wells must have the same number of samples"
        self.trap = trap
        self.wells = wells
        self.samples = wells[0].samples
        self.uopt = cx.Variable(shape=(self.samples, trap.num_electrodes), name="voltages")
        # self.offset = cx.Variable(shape=(self.samples, 1), name="offset")

    # def cost_x_potential(self, x, sample):
    # TODO this is wrong, find out why
    #     costs = []
    #     for well in self.wells:
    #         x1 = well.roi(x, sample)
    #         weight = well.weight(x1, sample)
    #         # here I evaluate the moments function over x[roi], I'd slice them if they were sampled instead
    #         moments = self.trap.eval_moments(x1)
    #         pot = well.potential(x1, sample)
    #         costs.append(
    #             cx.sum_squares(cx.multiply(np.sqrt(weight), self.uopt[sample] @ moments - pot))
    #         )
    #     return sum(costs)

    def cost_x_potential(self, x, sample):
        roi = np.zeros(x.shape, dtype=bool)
        for w in self.wells:
            roi += w.roi(x, sample)
        x1 = x[roi]
        moments = self.trap.eval_moments(x1)
        pot = np.sum([well.gaussian_potential(x1) for well in self.wells], axis=0)
        return cx.sum_squares(self.uopt[sample] @ moments - pot)

    def cost_hessian(self, sample):
        costs = []
        for w in self.wells:
            print("--- hessian cost one well")
            x = w.x0[sample]
            print(f"well at {x}")
            target_h = w.hessian[sample]
            with np.printoptions(suppress=True):
                print(curv_to_freq(target_h) * 1e-6)
            h_dc = self.trap.eval_hessian(x)
            print(h_dc.shape)
            h_ps = self.trap.pseudo_hessian(x)
            wh = np.einsum('i,ijk', self.uopt.value[sample], h_dc) + h_ps
            with np.printoptions(suppress=True):
                print(curv_to_freq(wh) * 1e-6)
            
            h_dc = h_dc.reshape(h_dc.shape[0], -1)
            ww = (self.uopt[sample] @ h_dc + h_ps.ravel() - target_h.ravel())
            print(ww.shape)
            costs.append(
                cx.sum_squares(cx.multiply(np.sqrt(h_weight), ww))
            )
        return sum(costs)

    def static_solver(self, x=None, default_V=None,
                      r0=1e-6, rx=1, rh=1e-3,
                      extra_constraints=[],
                      solver="MOSEK", verbose=False):
        """Static solver

        Args:
        moments

        Returns:
        uopt
        """
        num_electrodes = self.trap.num_electrodes
        x = self.trap.transport_axis if x is None else x
        default_V = self.trap.default_V if default_V is None else default_V

        # uopt = cx.Variable(num_electrodes, name="uopt_static")

        # static voltage cost
        cost = cx.sum_squares(cx.multiply(np.sqrt(r0), self.uopt - default_V))
        for j in range(self.samples):
            cost += cx.multiply(rx, self.cost_x_potential(x, j))
            if rh:
                cost += cx.multiply(rh, self.cost_hessian(j))

        # setup constrains
        constraints = [self.trap.min_V <= self.uopt, self.uopt <= self.trap.max_V]
        # TODO: handle extra constraints like y_sign
        # for j in range(num_electrodes // 2):
        #     constr += [uopt[j] == y_sign * uopt[10 + j]]

        constraints.extend(extra_constraints)

        objective = cx.Minimize(cost)
        problem = cx.Problem(objective, constraints)
        problem.solve(solver=solver, warm_start=True, verbose=verbose)
        return self.uopt
