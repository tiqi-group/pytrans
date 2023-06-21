#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import cvxpy as cx
from .objectives import Objective
from typing import List


from tqdm import tqdm
import multiprocessing
# import concurrent.futures


def process_obj_cstr(voltages, ci, trap):
    costs = []
    cstr = []
    for cj in ci:
        if cj.constraint_type is None:
            costs.extend(cj.objective(trap, voltages))
        else:
            cstr.extend(cj.constraint(trap, voltages))
    return costs, cstr


def init_waveform(n_samples, n_electrodes, name='Waveform'):
    return cx.Variable((n_samples, n_electrodes), name=name)


def solver(objectives: List[Objective],
           #    extra_constraints: List[Any] = None,
           #    trap_filter: Optional[TrapFilterTransform] = None,
           solver="MOSEK", verbose=True):
    """Static solver

        Args:
        step_objectives: list of lists of objectives
        global_objectives: list of objectives

        Returns:
        waveform: shape = (num_timesteps, num_electrodes)
        """

    # static voltage cost
    costs = []
    cstr = []

    objectives_iter = tqdm(objectives, desc="Compiling objectives") if verbose else objectives
    for obj in objectives_iter:
        if obj.constraint_type is None:
            costs.append(obj.objective())
        else:
            cstr.append(obj.constraint())

    cost = sum(costs)
    objective = cx.Minimize(cost)
    problem = cx.Problem(objective, cstr)

    if solver == "MOSEK":
        # mosek_params = {}
        mosek_params = {
            "MSK_IPAR_NUM_THREADS": multiprocessing.cpu_count(),
            # "MSK_IPAR_INFEAS_REPORT_AUTO": "MSK_ON"
        }

    problem.solve(solver=solver, warm_start=True, verbose=verbose, mosek_params=mosek_params)

    final_costs = []
    # for voltages, ci in zip(waveform, step_objectives):
    #     final_costs.append({
    #         f"{j}_{cj.__class__.__name__}": [c for c in cj.objective(trap, voltages)] for j, cj in enumerate(ci)
    #     })
    # final_costs.append({
    #     f"{j}_global_{cj.__class__.__name__}": [c for c in cj.objective(trap, waveform)] for j, cj in enumerate(global_objectives)
    # })
    results = {
        'problem': problem,
        'çost': cost,
        'final_costs': final_costs
    }

    return results
