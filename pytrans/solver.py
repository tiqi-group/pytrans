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
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import multiprocessing
# import concurrent.futures


def _process_objective(obj: Objective):
    if obj.constraint_type is None:
        res = obj.objective()
    else:
        res = obj.constraint()
    return obj.constraint_type, res


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

    # objectives_iter = tqdm(objectives, desc="Compiling objectives") if verbose else objectives
    # for obj in objectives_iter:
    #     if obj.constraint_type is None:
    #         costs.append(obj.objective())
    #     else:
    #         cstr.append(obj.constraint())
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(_process_objective, obj) for obj in objectives
        ]
        for future in tqdm(as_completed(futures), total=len(objectives), desc="Compiling objectives"):
            constraint_type, result = future.result()
            if constraint_type is None:
                costs.append(result)
            else:
                cstr.append(result)

    cost = sum(costs)
    objective = cx.Minimize(cost)
    problem = cx.Problem(objective, cstr)

    solver_kwargs = {}
    if solver == "MOSEK":
        # mosek_params = {}
        mosek_params = {
            "MSK_IPAR_NUM_THREADS": multiprocessing.cpu_count(),
            # "MSK_IPAR_INFEAS_REPORT_AUTO": "MSK_ON"
        }
        solver_kwargs['mosek_params'] = mosek_params

    problem.solve(solver=solver, warm_start=True, verbose=verbose, **solver_kwargs)

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
        'cost': cost,
        'final_costs': final_costs
    }

    return results
