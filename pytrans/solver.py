#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import cvxpy as cx
from .trap_model.abstract_trap import AbstractTrap
from .objectives import Objective
from typing import List


def solver(trap: AbstractTrap,
           objectives: List[List[Objective]],
           constraints: List[Objective],
           electrode_indices=None,
           solver="MOSEK", verbose=False):
    """Static solver

        Args:
        objectives: list of lists of objectives
        constraints: list of contstraints

        Returns:
        waveform: shape = (num_timesteps, num_electrodes)
        """

    # static voltage cost
    num_electrodes = trap.num_electrodes if electrode_indices is None else len(electrode_indices)
    electrode_indices = slice(None) if electrode_indices is None else electrode_indices
    shape = (len(objectives), num_electrodes)
    print(shape)
    waveform = cx.Variable(shape=shape, name="waveform")
    costs = []
    cstr = []

    for voltages, ci in zip(waveform, objectives):
        for cj in ci:
            if cj.constraint_type is None:
                costs.extend(cj.objective(trap, voltages, electrode_indices))
            else:
                cstr.extend(cj.constraint(trap, voltages, electrode_indices))

    for c in constraints:
        cstr.extend(c.constraint(trap, waveform, electrode_indices))

    cost = sum(costs)
    objective = cx.Minimize(cost)
    problem = cx.Problem(objective, cstr)
    problem.solve(solver=solver, warm_start=True, verbose=verbose)

    final_costs = []
    for voltages, ci in zip(waveform, objectives):
        final_costs.append({f"{j}_{cj.__class__.__name__}": [c.value for c in cj.objective(trap, voltages, electrode_indices)] for j, cj in enumerate(ci)})

    return waveform, final_costs
