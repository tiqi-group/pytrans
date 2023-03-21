#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np
from typing import Union, List, Dict
from nptyping import NDArray

from pytrans.typing import Coords, Coords1, Roi, Bounds

from pytrans.ions import Ion
from pytrans.timer import timer
from pytrans.abstract_model import AbstractTrapModel

from pytrans.plotting.plotting import plot3d_make_layout, plot3d_potential
from .mode_solver import mode_solver
from .results import AnalysisResults

from scipy.optimize import minimize

# from itertools import permutations


__roi: Roi = np.asarray((400e-6, 30e-6, 30e-6))
__default_minimize_options = dict(accuracy=1e-8)


# def _eig_hessian(H, sort_close_to=None):
#     h, vs = np.linalg.eig(H)
#     if sort_close_to is not None:
#         ix = _sort_close_to(h, sort_close_to)
#     else:
#         ix = [np.argmax(abs(vs[0])), np.argmax(abs(vs[1])), np.argmax(abs(vs[2]))]
#     # ix = np.argsort(abs(h))
#     h = h[ix]
#     vs = vs[:, ix]
#     angle = np.arctan2(1, vs[1, 2] / vs[2, 2]) * 180 / np.pi
#     return h, vs, angle


# def _sort_close_to(x0, x1):
#     perm = list(map(list, permutations(range(len(x1)))))
#     diff = np.asarray([x0 - x1[s] for s in perm])
#     ix = np.argmin((diff**2).sum(1))
#     return perm[ix]


def _bounds_from_roi(r0: Coords1, roi: Roi) -> Bounds:
    return [(-r + x, r + x) for x, r in zip(r0, roi)]


def _minimize_potential_single_ion(trap: AbstractTrapModel, voltages: NDArray, ion: Ion,
                                   r0: Coords1, bounds: Bounds, pseudo: bool,
                                   minimize_options: Dict, verbose: bool):

    def fun3(xyz):
        return trap.potential(voltages, *xyz, ion.mass_amu, pseudo=pseudo)

    def jac3(xyz):
        return trap.gradient(voltages, *xyz, ion.mass_amu, pseudo=pseudo)

    _minimize = timer(minimize) if verbose else minimize
    res = _minimize(fun3, r0, method='TNC', jac=jac3, bounds=bounds, options=minimize_options)

    return res


def _analyse_potential_single_ion(trap: AbstractTrapModel, voltages: NDArray, ion: Ion,
                                  r0: Coords1, bounds: Bounds, pseudo: bool,
                                  find_3dmin: bool, minimize_options: Dict,
                                  verbose: bool, title: str):

    if find_3dmin:
        # TODO should use the mode solver directly here? Something like
        # res = mode_solver(trap, voltages, ions=ion, x0=np.asarray(r0).reshape(1, -1), bounding_box=bounds)
        minimize_results = _minimize_potential_single_ion(trap, voltages, ion, r0, bounds, pseudo,
                                                          minimize_options, verbose)
        x_eq = minimize_results.x
    else:
        minimize_results = None
        x_eq = r0

    fun = trap.potential(voltages, x_eq[0], x_eq[1], x_eq[2], ion.mass_amu, pseudo=pseudo)
    jac = trap.gradient(voltages, x_eq[0], x_eq[1], x_eq[2], ion.mass_amu, pseudo=pseudo)
    hess = trap.hessian(voltages, x_eq[0], x_eq[1], x_eq[2], ion.mass_amu, pseudo=pseudo)

    results = AnalysisResults(ion, x_eq, fun, jac, hess, minimize_results, title=title, mode_solver_results=None)

    return results


def analyse_potential(trap: AbstractTrapModel, voltages: NDArray, ions: Union[Ion, List[Ion]],
                      r0: Union[Coords1, Coords], find_3dmin=True, pseudo=True,
                      plot=True, axes=None, title='',
                      roi=None, minimize_options=dict(), verbose=True):

    r0 = np.asarray(r0)
    if isinstance(ions, Ion):
        ion1 = ions
        r_cm = r0
        _run_mode_solver = False  # is there a better way to do this?
    else:
        assert r0.ndim == 2 and r0.shape[0] == len(ions)
        avg_mass_amu = np.asarray([_ion.mass_amu for _ion in ions]).mean()
        ion1 = Ion(f"Average{ions}", mass_amu=avg_mass_amu, unit_charge=1)  # TODO fix this to charge > 1
        r_cm = r0.mean(axis=0)
        _run_mode_solver = True

    roi = __roi if roi is None else roi
    bounds = _bounds_from_roi(r_cm, roi)
    _minimize_options = __default_minimize_options.copy()
    _minimize_options.update(minimize_options)
    results = _analyse_potential_single_ion(trap, voltages, ion1, r_cm, bounds, pseudo,
                                            find_3dmin, _minimize_options, verbose, title)

    if _run_mode_solver:
        res = mode_solver(trap, voltages, ions, r0, bounds, _minimize_options)
        results.mode_solver_results = res
    else:
        results.mode_solver_results = None

    if verbose:
        print(results)

    if not plot:
        return results

    if axes is None:
        fig, axes = plot3d_make_layout(n=1)

    fig, axes = plot3d_potential(trap, voltages, ion1, r_cm, roi, axes=axes, pseudo=pseudo, analyse_results=results, title=title)

    # res['fig'] = fig
    # res['axes'] = axes

    return results
