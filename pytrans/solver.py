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

from .conversion import curv_to_freq, C, E0
from .utils.timer import timer

import logging
logger = logging.getLogger(__name__)

# h_weight = np.asarray([
#     [1, 1e-3, 1e-3],
#     [1e-3, 1, 1],
#     [1e-3, 1, 1]
# ]).ravel()
# h_weight = np.ones((9,))
# h_weight[0] = 0


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
    
    @timer
    def cost_x_potential_q(self, x, sample):
        roi = np.zeros(x.shape, dtype=bool)
        for w in self.wells:
            # build union of all the rois
            roi += w.roi(x, sample)
        x1 = x[roi]
        moments = self.trap.eval_moments(x1)
        # offset = cx.Variable((1,))
        costs = []
        for well in self.wells:
            weight = well.weight(x1, sample)
            # here I evaluate the moments function over x[roi], I'd slice them if they were sampled instead
            pot = well.potential(x1, sample)
            diff = (self.uopt[sample] @ moments - pot)
            costs.append(
                cx.sum_squares(cx.multiply(np.sqrt(weight), diff))
            )
        return sum(costs)

    @timer
    def cost_x_potential_g(self, x, sample):
        # g like gaussian
        # this does not make use of the ROI since the gaussian is anyway finite
        moments = self.trap.eval_moments(x)
        pot = np.sum([well.gaussian_potential(x, sample) for well in self.wells], axis=0)
        weight = np.sum([well.weight(x, sample) for well in self.wells], axis=0)
        offset = cx.Variable((1,))
        diff = (self.uopt[sample] @ moments - pot - offset) / max(self.wells[0].depth)
        # return cx.sum_squares(cx.multiply(np.sqrt(weight), diff))
        return cx.sum_squares(diff)

    @timer
    def cost_hessian(self, sample):
        costs = []
        for w in self.wells:
            print("--- hessian cost one well")
            x = w.x0[sample]
            print(f"Hessian for well at {x}")
            target_h = w.hessian[sample]
            with np.printoptions(suppress=True):
                print(curv_to_freq(target_h) * 1e-6)
            h_dc = self.trap.eval_hessian(x)
            print(h_dc.shape)
            h_ps = self.trap.pseudo_hessian(x)

            # test
            if self.uopt.value is not None:
                with np.printoptions(suppress=True):
                    wh = np.einsum('i,ijk', self.uopt.value[sample], h_dc) + h_ps
                    print(curv_to_freq(wh) * 1e-6)

            h_dc = h_dc.reshape(h_dc.shape[0], -1)
            ww = (self.uopt[sample] @ h_dc + h_ps.ravel() - target_h.ravel()) / C * 1e-12
            print(ww.shape)
            costs.append(
                cx.sum_squares(ww)
                # cx.sum_squares(cx.multiply(np.sqrt(h_weight), ww))
            )
        return sum(costs)

    @timer
    def cost_hessian_dc(self, sample):
        costs = []
        for w in self.wells:
            print("--- hessian cost one well")
            x = w.x0[sample]
            print(f"Hessian DC for well at {x}")
            target_h = w.hessian_dc[sample]
            with np.printoptions(suppress=True):
                print(curv_to_freq(target_h) * 1e-6)
            h_dc = self.trap.eval_hessian(x)

            h_dc = h_dc.reshape(h_dc.shape[0], -1)
            ww = (self.uopt[sample] @ h_dc - target_h.ravel()) / C * 1e-12
            print(ww.shape)
            costs.append(
                cx.sum_squares(ww)
                # cx.sum_squares(cx.multiply(np.sqrt(h_weight), ww))
            )
        return sum(costs)

    @timer
    def cost_gradient(self, sample):
        costs = []
        for w in self.wells:
            print("--- gradient cost one well")
            x = w.x0[sample]
            print(f"field gradient for well at {x}")

            e_dc = self.trap.eval_gradient(x)  # (ele, 3)
            e_ps = self.trap.pseudo_gradient(x)

            if self.uopt.value is not None:
                with np.printoptions(suppress=True):
                    eh = self.uopt.value[sample] @ e_dc
                    print(eh, E0)
                    print(eh / E0)

            ee = (self.uopt[sample] @ e_dc + e_ps) / E0
            print(ee.shape)
            costs.append(cx.sum_squares(ee))
        return sum(costs)

    @timer
    def cost_slew0(self):
        # gradient matrix
        M = np.zeros((self.samples, self.samples))
        M += np.diagflat([1] * (self.samples - 1), k=1) / 2
        M += np.diagflat([-1] * (self.samples - 1), k=-1) / 2
        M[0, [0, 1]] = -1, 1
        M[-1, [-2, -1]] = -1, 1
        print("--- grad")
        print((M @ self.uopt).shape)
        return cx.sum_squares(M @ self.uopt)

    @timer
    def solver(self, x=None,
               method_x='g',
               rx=1, rh=1, r0=0,
               rd=1,
               start_V=None, stop_V=None,
               default_V=None,
               extra_constraints=[],
               solver="MOSEK", verbose=False):
        """Static solver

        Args:
        moments

        Returns:
        uopt
        """
        x = self.trap.transport_axis if x is None else x
        default_V = self.trap.default_V if default_V is None else default_V

        # static voltage cost
        cost = 0
        assert method_x in ['q', 'g']
        cost_x = getattr(self, f"cost_x_potential_{method_x}")
        for j in range(self.samples):
            if rx:
                cost += cx.multiply(rx, cost_x(x, j))
            if rh:
                cost += cx.multiply(rh, self.cost_hessian(j) + self.cost_gradient(j))
        if r0:
            cost += cx.multiply(r0, cx.sum_squares(self.uopt - default_V))

        if self.samples > 1:
            if r0:
                cost += cx.multiply(rd, self.cost_slew0())

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
