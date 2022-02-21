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

from tqdm import tqdm


def solver(trap: AbstractTrap,
           step_objectives: List[List[Objective]],
           global_objectives: List[Objective],
           extra_constraints: List[Any] = None,
           solver="MOSEK", start_value=None, verbose=False):
    """Static solver

        Args:
        step_objectives: list of lists of objectives
        global_objectives: list of objectives

        Returns:
        waveform: shape = (num_timesteps, num_electrodes)
        """

    # static voltage cost
    shape = (len(step_objectives), trap.n_electrodes)
    waveform = cx.Variable(shape=shape, name="waveform")
    costs = []  # type: ignore
    cstr = []   # type: ignore

    for voltages, ci in zip(waveform, tqdm(step_objectives, desc="Compiling step objectives")):
        for cj in ci:
            if cj.constraint_type is None:
                costs.extend(cj.objective(trap, voltages))
            else:
                cstr.extend(cj.constraint(trap, voltages))

    for c in global_objectives:
        if c.constraint_type is None:
            costs.extend(c.objective(trap, waveform))
        else:
            cstr.extend(c.constraint(trap, waveform))

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

    return waveform, final_costs
