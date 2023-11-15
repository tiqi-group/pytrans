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
from dataclasses import dataclass

from tqdm import tqdm
from multiprocessing import cpu_count


def _process_objective(obj: Objective):
    if obj.constraint_type is None:
        res = obj.objective()
    else:
        res = obj.constraint()
    return obj.constraint_type, res


def init_waveform(n_samples, n_electrodes, name='waveform'):
    return cx.Variable((n_samples, n_electrodes), name=name)


@dataclass(frozen=True)
class SolverResults:
    problem: cx.Problem
    variables: dict[str, cx.Variable]
    costs: list[cx.Expression]
    constraints: list[cx.Constraint]
    waveform: cx.Variable | None


def _compile_objectives(objectives: List[Objective], verbose=True, parallel_compile=False):
    # static voltage cost
    costs = []
    constraints = []

    if parallel_compile:
        with ProcessPoolExecutor(max_workers=None) as executor:
            futures = [
                executor.submit(_process_objective, obj) for obj in objectives
            ]
            futures_iter = tqdm(as_completed(futures), total=len(objectives), desc="Compiling objectives") if verbose else as_completed(futures)
            for future in futures_iter:
                constraint_type, result = future.result()
                if constraint_type is None:
                    costs.append(result)
                else:
                    constraints.append(result)
    else:
        objectives_iter = tqdm(objectives, desc="Compiling objectives") if verbose else objectives
        for obj in objectives_iter:
            if obj.constraint_type is None:
                costs.append(obj.objective())
            else:
                constraints.append(obj.constraint())

    return costs, constraints


def _solve(costs: list, constraints: list, solver="MOSEK", verbose=True, solver_kwargs={}):
    cost = cx.sum(costs)
    objective = cx.Minimize(cost)
    problem = cx.Problem(objective, constraints)

    _kwargs = {}
    if solver == "MOSEK":
        mosek_params = {
            "MSK_IPAR_NUM_THREADS": cpu_count(),
            # "MSK_IPAR_INFEAS_REPORT_AUTO": "MSK_ON"
        }
        _kwargs['mosek_params'] = mosek_params
    _kwargs.update(solver_kwargs)
    problem.solve(solver=solver, warm_start=False, verbose=verbose, **_kwargs)

    variables = dict()
    for v in problem.variables():
        # variables that have been copied by multiprocessing actually have the same name and value
        variables[v.name()] = v
    waveform = variables.get('waveform', None)
    results = SolverResults(problem, variables, costs, constraints, waveform)
    return results


def solver(objectives: List[Objective],
           solver="MOSEK", verbose=True, parallel_compile=False) -> SolverResults:
    """Waveform solver

        Args:
        objectives: list of objectives

        Returns:
        results: an object containing the optimization results
    """
    costs, constraints = _compile_objectives(objectives, verbose, parallel_compile)
    results = _solve(costs, constraints, solver, verbose)
    return results
