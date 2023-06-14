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
import multiprocessing
import concurrent.futures



def process_obj_cstr(voltages, ci, trap):
    costs = []
    cstr = []
    for cj in ci:
        if cj.constraint_type is None:
            costs.extend(cj.objective(trap, voltages))
        else:
            cstr.extend(cj.constraint(trap, voltages))
    return costs, cstr

def solver(trap: AbstractTrap,
           step_objectives: List[List[Objective]],
           global_objectives: List[Objective],
           extra_constraints: List[Any] = None,
           trap_filter=None,
           solver="MOSEK", start_value=None, verbose=True):
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
        print('watch out, filter in this modele is overriden here-----------------------------')
        from tools.cryo_filters import cryo_filter_without_amp_filter

        b, a, dt = cryo_filter_without_amp_filter
        _t, h = sg.dimpulse((b, a, dt))
        h = np.squeeze(h)
        m = len(h) - 1
        n_eval_after = 50
        waveform0 = cx.Variable(shape=(n_steps + m + n_eval_after, trap.n_electrodes), name="waveform")
        for j in range(m):
            cstr.append(waveform0[j] == waveform0[m])
        for j in range(n_eval_after):
            cstr.append(waveform0[m+n_steps+j] == waveform0[m+n_steps-1])
        M = convolution_matrix(h, waveform0.shape[0], mode='valid')
        waveform = M @ waveform0
        waveform0 = waveform0[m:]
        waveform0 = waveform0[:-n_eval_after]
        for _ in range(n_eval_after):
            step_objectives.append(step_objectives[-1])
        step_objectives[-1][0].weight*=10
        

    print("waveform.shape[0]",waveform.shape[0])
    print("len(step_objective)",len(step_objectives))

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future_to_result = {
                executor.submit(
                    process_obj_cstr,
                    voltage,
                    ci,
                    trap
                ) for (voltage, ci) in zip(waveform, step_objectives)}
            for future in tqdm(concurrent.futures.as_completed(future_to_result), total=waveform.shape[0], desc="Compiling step objectives"):
                cost, cstr_res = future.result()
                costs.extend(cost)
                cstr.extend(cstr_res)
    except Exception as e:
        print(e)
        print("Error in parallel processing. Switching to sequential")
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

    if solver=="MOSEK":
        # mosek_params = {}
        mosek_params = {
            "MSK_IPAR_NUM_THREADS":multiprocessing.cpu_count(),
            "MSK_IPAR_INFEAS_REPORT_AUTO":"MSK_ON"
        }
        
    problem.solve(solver=solver, warm_start=True, verbose=verbose, mosek_params=mosek_params)

    final_costs = []
    # for voltages, ci in zip(waveform, step_objectives):
    #     final_costs.append({f"{j}_{cj.__class__.__name__}": [c for c in cj.objective(trap, voltages)] for j, cj in enumerate(ci)})
    # final_costs.append({f"{j}_global_{cj.__class__.__name__}": [c for c in cj.objective(trap, waveform)] for j, cj in enumerate(global_objectives)})

    return waveform0, final_costs
