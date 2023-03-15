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

from pytrans.typing import Coords, Coords1, Roi

from pytrans.ions import Ion
from pytrans.timer import timer
from pytrans.abstract_model import AbstractTrapModel

from pytrans.plotting.plotting import plot3d_make_layout, plot3d_potential
from .mode_solver import mode_solver
from .analysis_results import AnalysisResults, _color_str, Fore

from scipy.optimize import minimize

from itertools import permutations


__roi: Roi = np.asarray((400e-6, 30e-6, 30e-6))
__default_minimize_options = dict(accuracy=1e-8)


def _eig_hessian(H, sort_close_to=None):
    h, vs = np.linalg.eig(H)
    if sort_close_to is not None:
        ix = _sort_close_to(h, sort_close_to)
    else:
        ix = [np.argmax(abs(vs[0])), np.argmax(abs(vs[1])), np.argmax(abs(vs[2]))]
    # ix = np.argsort(abs(h))
    h = h[ix]
    vs = vs[:, ix]
    angle = np.arctan2(1, vs[1, 2] / vs[2, 2]) * 180 / np.pi
    return h, vs, angle


def _sort_close_to(x0, x1):
    perm = list(map(list, permutations(range(len(x1)))))
    diff = np.asarray([x0 - x1[s] for s in perm])
    ix = np.argmin((diff**2).sum(1))
    return perm[ix]


def _bounds_from_roi(r0: Coords1, roi: Roi):
    return [(-r + x, r + x) for x, r in zip(r0, roi)]


def _minimize_potential_single_ion(trap: AbstractTrapModel, voltages: NDArray, ion: Ion,
                                   r0: Coords1, roi: Roi, pseudo: bool,
                                   minimize_options: Dict, verbose: bool):

    def fun3(xyz):
        return trap.potential(voltages, *xyz, ion.mass_amu, pseudo=pseudo)

    def jac3(xyz):
        return trap.gradient(voltages, *xyz, ion.mass_amu, pseudo=pseudo)

    bounds = _bounds_from_roi(r0, roi)

    opts = __default_minimize_options.copy()
    opts.update(minimize_options)
    _minimize = timer(minimize) if verbose else minimize
    res = _minimize(fun3, r0, method='TNC', jac=jac3, bounds=bounds, options=opts)

    return res


def _analyse_potential_single_ion(trap: AbstractTrapModel, voltages: NDArray, ion: Ion,
                                  r0: Coords1, roi: Roi, pseudo: bool,
                                  find_3dmin: bool, minimize_options: Dict,
                                  verbose: bool, title: str):

    if find_3dmin:
        # TODO should use the mode solver directly here? Something like
        # res = mode_solver(trap, voltages, ions=ion, x0=np.asarray(r0).reshape(1, -1), bounding_box=bounds)
        minimize_result = _minimize_potential_single_ion(trap, voltages, ion, r0, roi, pseudo,
                                                         minimize_options, verbose)
        x_eq = minimize_result.x
    else:
        if verbose:
            print(_color_str(Fore.YELLOW, "Set position to r0"))
        minimize_result = None
        x_eq = r0

    fun = trap.potential(voltages, x_eq[0], x_eq[1], x_eq[2], ion.mass_amu, pseudo=pseudo)
    jac = trap.gradient(voltages, x_eq[0], x_eq[1], x_eq[2], ion.mass_amu, pseudo=pseudo)
    hess = trap.hessian(voltages, x_eq[0], x_eq[1], x_eq[2], ion.mass_amu, pseudo=pseudo)

    # h, vs, angle = _eig_hessian(H)
    # freqs = curv_to_freq(h, ion=ion)
    # shift = field_to_shift(E, ion=ion)

    result = AnalysisResults(ions=[ion], x0=np.atleast_2d(r0), x_eq=np.atleast_2d(x_eq),
                             fun=fun, jac=jac, hess=hess, minimize_result=minimize_result)

    return result


def analyse_potential(trap: AbstractTrapModel, voltages: NDArray, ions: Union[Ion, List[Ion]],
                      r0: Union[Coords1, Coords], find_3dmin=True, pseudo=True,
                      plot=True, axes=None, title='',
                      roi=None, minimize_options=dict(), verbose=True):

    roi = __roi if roi is None else roi

    ions = [ions] if isinstance(ions, Ion) else ions

    if len(ions) == 1:
        if verbose:
            print(_color_str(Fore.YELLOW, f"--------------\nAnalyse potential for ion {ions[0]}: {title}"))
        res = _analyse_potential_single_ion(trap, voltages, ions[0], r0, roi, pseudo,
                                            find_3dmin, minimize_options, verbose, title)
    else:
        if verbose:
            print(_color_str(Fore.YELLOW, f"--------------\nAnalyse potential for ion string {ions}: {title}"))
        bounds = _bounds_from_roi(r0.mean(axis=0), roi)
        res = mode_solver(trap, voltages, ions, x0=r0, bounding_box=bounds, minimize_options=minimize_options)

    if verbose:
        print(res)

    # plot = False
    if not plot:
        return res

    if axes is None:
        fig, axes = plot3d_make_layout(n=1)

    fig, axes = plot3d_potential(trap, voltages, ions, r0, analyse_results=res, roi=roi, axes=axes, pseudo=pseudo, title=title)

    # res['fig'] = fig
    # res['axes'] = axes

    return res
