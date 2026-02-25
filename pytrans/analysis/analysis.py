#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

"""
Module docstring
"""

import numpy as np
from tqdm import tqdm
from typing import Union, List, Dict, Optional
from numpy.typing import NDArray

from pytrans.typing import Coords, Coords1, RoiSize, Bounds, Waveform
from pytrans.plotting.roi import Roi

from pytrans.ions import Ion
from pytrans.timer import timer
from pytrans.abstract_model import AbstractTrapModel

from pytrans.plotting.plotting import plot_potential
from .mode_solver import mode_solver, init_crystal, diagonalize_hessian
from .results import AnalysisResults

from scipy.optimize import minimize

from concurrent.futures import ProcessPoolExecutor, as_completed

# from itertools import permutations


__default_minimize_options = dict(accuracy=1e-8)


def _minimize_potential_single_ion(
    trap: AbstractTrapModel,
    voltages: NDArray,
    ion: Ion,
    r0: Coords1,
    bounds: Bounds,
    pseudo: bool,
    minimize_options: Dict,
    verbose: bool,
):
    def fun3(xyz):
        return trap.potential(voltages, *xyz, ion.mass_amu, pseudo=pseudo)

    def jac3(xyz):
        return trap.gradient(voltages, *xyz, ion.mass_amu, pseudo=pseudo)

    _minimize = timer(minimize) if verbose else minimize
    res = _minimize(fun3, r0, method="TNC", jac=jac3, bounds=bounds, options=minimize_options)

    return res


def _analyse_potential_single_ion(
    trap: AbstractTrapModel,
    voltages: NDArray,
    ion: Ion,
    r0: Coords1,
    bounds: Bounds,
    pseudo: bool,
    find_3dmin: bool,
    minimize_options: Dict,
    verbose: bool,
    title: str,
):
    if find_3dmin:
        r0 += np.random.randn(*r0.shape) * 1e-8
        minimize_results = _minimize_potential_single_ion(
            trap, voltages, ion, r0, bounds, pseudo, minimize_options, verbose
        )
        x_eq = minimize_results.x
    else:
        minimize_results = None
        x_eq = r0

    fun = trap.potential(voltages, x_eq[0], x_eq[1], x_eq[2], ion.mass_amu, pseudo=pseudo)
    jac = trap.gradient(voltages, x_eq[0], x_eq[1], x_eq[2], ion.mass_amu, pseudo=pseudo)
    hess = trap.hessian(voltages, x_eq[0], x_eq[1], x_eq[2], ion.mass_amu, pseudo=pseudo)

    mode_freqs, mode_vectors = diagonalize_hessian([ion], hess)
    mode_vectors = np.squeeze(mode_vectors)
    results = AnalysisResults(
        ion,
        x_eq,
        fun,
        jac,
        hess,
        mode_freqs,
        mode_vectors,
        minimize_results=minimize_results,
        title=title,
        mode_solver_results=None,
    )

    return results


def analyse_potential(
    trap: AbstractTrapModel,
    voltages: NDArray,
    ions: Union[Ion, List[Ion]],
    r0: Union[Coords1, Coords],
    roi: Union[RoiSize, Bounds],
    ion1: Optional[Ion] = None,
    find_3dmin=True,
    pseudo=True,
    plot=True,
    plot_roi=None,
    trap_axis="x",
    axes=None,
    title="",
    minimize_options=dict(),
    verbose=True,
):
    r0 = np.asarray(r0, dtype=float)
    if isinstance(ions, Ion):
        ion1 = ions
        r_cm = r0
        _run_mode_solver = False  # is there a better way to do this?
        _axis = None
    else:
        if r0.ndim == 1:
            r_cm = r0
            _axis = "xyz".index(trap_axis)
            r0 = init_crystal(r0, dx=5e-6, n_ions=len(ions), axis=_axis)
        else:
            r_cm = r0.mean(axis=0)
            _axis = None
        if ion1 is None:
            avg_mass_amu = np.asarray([_ion.mass_amu for _ion in ions]).mean()
            ion1 = Ion(f"Average{ions}", mass_amu=avg_mass_amu, unit_charge=1)  # TODO fix this to charge > 1
        _run_mode_solver = True

    roi = Roi(roi, center=r_cm)
    bounds = roi.bounds
    _minimize_options = __default_minimize_options.copy()
    _minimize_options.update(minimize_options)
    results = _analyse_potential_single_ion(
        trap, voltages, ion1, r_cm, bounds, pseudo, find_3dmin, _minimize_options, verbose, title
    )

    if _run_mode_solver:
        res = mode_solver(trap, voltages, ions, r0, bounds, _axis, _minimize_options)
        results.mode_solver_results = res
    else:
        results.mode_solver_results = None

    if verbose:
        print(results)

    if plot:
        r0 = results.x_eq
        plot_roi = roi._size if plot_roi is None else plot_roi
        plot_potential(
            trap,
            voltages,
            ion1,
            r0,
            plot_roi,
            trap_axis=trap_axis,
            axes=axes,
            pseudo=pseudo,
            analyse_results=results,
            title=title,
        )

    return results


def analyse_waveform(
    trap: AbstractTrapModel,
    waveform: Waveform,
    ions: Union[Ion, List[Ion]],
    r0: Union[Coords1, Coords],
    roi: Union[RoiSize, Bounds],
    ion1: Optional[Ion] = None,
    find_3dmin=True,
    pseudo=True,
    title="",
    minimize_options=dict(),
    max_workers=None,
):
    results = []
    _kwargs = dict(
        trap=trap,
        ions=ions,
        ion1=ion1,
        find_3dmin=find_3dmin,
        pseudo=pseudo,
        title=title,
        roi=roi,
        minimize_options=minimize_options,
        plot=False,
        verbose=False,
    )
    L = len(waveform)
    r0 = np.asarray(r0)
    r0s = r0 if r0.ndim > 1 else [r0] * L
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_result = {
            executor.submit(analyse_potential, voltages=waveform[j], r0=r0s[j], **_kwargs): j for j in range(L)
        }
        for future in tqdm(as_completed(future_to_result), total=len(waveform), desc=f"Waveform analysis: {title}"):
            j = future_to_result[future]
            try:
                result = future.result()
                results.append((result, j))
            except Exception as exc:
                print(type(exc), exc)
                raise exc
    results.sort(key=lambda x: x[1])
    return [x[0] for x in results]
