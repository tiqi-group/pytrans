#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

"""
Module docstring
"""
import cvxpy as cx
from .objectives import Objective
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


def init_waveform(n_samples: int, n_electrodes: int, name: str = "waveform"):
    return cx.Variable((n_samples, n_electrodes), name=name)


@dataclass(frozen=True)
class SolverResults:
    problem: cx.Problem
    variables: dict[str, cx.Variable]
    costs: list[cx.Expression]
    constraints: list[cx.Constraint]
    waveform: cx.Variable | None


def _compile_objectives(objectives: list[Objective], verbose: bool, parallel_compile: bool):
    # static voltage cost
    costs = []
    constraints = []

    if parallel_compile:
        with ProcessPoolExecutor(max_workers=None) as executor:
            futures = [executor.submit(_process_objective, obj) for obj in objectives]
            futures_iter = (
                tqdm(
                    as_completed(futures),
                    total=len(objectives),
                    desc="Compiling objectives",
                )
                if verbose
                else as_completed(futures)
            )
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


_specific_solver_options = {
    "MOSEK": {
        "mosek_params": {
            "MSK_IPAR_NUM_THREADS": cpu_count(),
            # "MSK_IPAR_INFEAS_REPORT_AUTO": "MSK_ON"
        }
    }
}


def _solve(
    costs: list,
    constraints: list,
    solver: str | None,
    verbose: bool,
    **solve_kwargs,
):
    cost = cx.sum(costs)
    objective = cx.Minimize(cost)
    problem = cx.Problem(objective, constraints)

    _kwargs = {}
    if solver is not None and solver in _specific_solver_options:
        _solver_options = _specific_solver_options[solver]
        _kwargs.update(_solver_options)
    _kwargs.update(solve_kwargs)
    problem.solve(solver=solver, warm_start=False, verbose=verbose, **_kwargs)

    variables = {}
    for v in problem.variables():
        # variables that have been copied by multiprocessing actually have the same name and value
        variables[v.name()] = v
    waveform = variables.get("waveform", None)
    results = SolverResults(problem, variables, costs, constraints, waveform)
    return results


def solver(
    objectives: list[Objective],
    solver: str | None = None,
    verbose: bool = True,
    parallel_compile: bool = False,
    **solve_kwargs,
) -> SolverResults:
    """Waveform solver

    Args:
    objectives: list of objectives

    Returns:
    results: an object containing the optimization results
    """
    costs, constraints = _compile_objectives(objectives, verbose, parallel_compile)
    results = _solve(costs, constraints, solver, verbose, **solve_kwargs)
    return results
