#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import cvxpy as cx
from .abstract_model import AbstractTrap
from .objectives import Objective
from typing import List, Any

import numpy as np
from scipy import signal as sg
from scipy.linalg import convolution_matrix

from tqdm import tqdm


def solver(trap: AbstractTrap,
           step_objectives: List[List[Objective]],
           global_objectives: List[Objective],
           extra_constraints: List[Any] = None,
           trap_filter=None,
           solver="MOSEK", start_value=None, verbose=False):
    """Static solver

        Args:
        step_objectives: list of lists of objectives
        global_objectives: list of objectives

        Returns:
        waveform: shape = (num_timesteps, num_electrodes)
        """

    # static voltage cost
    n_steps = len(step_objectives)
    costs = []
    cstr = []
    if trap_filter is None:
        waveform = cx.Variable(shape=(n_steps, trap.n_electrodes), name="waveform")
        waveform0 = waveform
    else:
        # this might be moved in the .trap_filter module
        b, a, dt = trap_filter
        _t, h = sg.dimpulse((b, a, dt))
        h = np.squeeze(h)
        m = len(h) - 1
        waveform0 = cx.Variable(shape=(n_steps + m, trap.n_electrodes), name="waveform")
        for j in range(m):
            cstr.append(waveform0[j] == waveform0[m])
        M = convolution_matrix(h, waveform0.shape[0], mode='valid')
        waveform = M @ waveform0
        waveform0 = waveform0[m:]

    step_iter = tqdm(step_objectives, desc="Compiling step objectives") if verbose else step_objectives
    for voltages, ci in zip(waveform, step_iter):
        for cj in ci:
            if cj.constraint_type is None:
                costs.extend(cj.objective(trap, voltages))
            else:
                cstr.extend(cj.constraint(trap, voltages))

    for c in global_objectives:
        # if c.__class__.__name__ == "SlewRateObjective":
        #     costs.extend(c.objective(trap, waveform0))
        # else:
        if c.constraint_type is None:
            costs.extend(c.objective(trap, waveform0))
        else:
            cstr.extend(c.constraint(trap, waveform0))

    if extra_constraints is not None:
        cstr.extend(extra_constraints)
    cost = sum(costs)
    objective = cx.Minimize(cost)
    problem = cx.Problem(objective, cstr)
    if start_value is not None:
        waveform.value = start_value
    problem.solve(solver=solver, warm_start=True, verbose=verbose)

    final_costs = []
    # for voltages, ci in zip(waveform, step_objectives):
    #     final_costs.append({f"{j}_{cj.__class__.__name__}": [c for c in cj.objective(trap, voltages)] for j, cj in enumerate(ci)})
    # final_costs.append({f"{j}_global_{cj.__class__.__name__}": [c for c in cj.objective(trap, waveform)] for j, cj in enumerate(global_objectives)})

    return waveform0, final_costs
