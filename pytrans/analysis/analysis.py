#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np
from typing import Any
from nptyping import NDArray, Shape
from pytrans.plotting import plot3d_potential, plot3d_make_layout, plot_fields_curvatures
from pytrans.conversion import curv_to_freq, field_to_shift
from pytrans.ions import Ion
from pytrans.timer import timer
from pytrans.abstract_model import AbstractTrapModel

from scipy.optimize import minimize

from itertools import permutations
from colorama import init as colorama_init, Fore
from tqdm import tqdm

colorama_init(autoreset=True)

__roi = (400, 30, 30)


def _color_str(color, str):
    return f"{color}{str:s}{Fore.RESET}"


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


def find_3dmin_potential(trap: AbstractTrapModel, voltages: NDArray, ion: Ion,
                         r0: NDArray[Shape["3"], Any], roi=None, pseudo=True,
                         minimize_options=dict(), verbose=True):
    roi = __roi if roi is None else roi

    def fun3(xyz):
        return trap.potential(voltages, *xyz, ion.mass_amu, pseudo=pseudo)

    def jac3(xyz):
        return trap.gradient(voltages, *xyz, ion.mass_amu, pseudo=pseudo)

    _roi = []
    for lim in roi:
        lim = lim if isinstance(lim, (int, float)) else min(lim)
        _roi.append(lim)

    bounds = [(-r * 1e-6 + x, r * 1e-6 + x) for r, x in zip(_roi, r0)]

    # res = mode_solver(trap, voltages, ions=ion, x0=np.asarray(r0).reshape(1, -1), bounding_box=bounds)
    opts = dict(accuracy=1e-8)
    opts.update(minimize_options)
    _minimize = timer(minimize) if verbose else minimize
    res = _minimize(fun3, r0, method='TNC', jac=jac3, bounds=bounds, options=opts)

    if verbose:
        print(_color_str(Fore.YELLOW, "Potential mimimum [um]"))
        print(res.x * 1e6)
        # print((res.x - r0) * 1e6)
    # return res.x_eq.ravel()
    return res.x


def analyse_potential_data(trap: AbstractTrapModel, voltages: NDArray, ion: Ion,
                           r0: NDArray[Shape["3"], Any], roi=None, pseudo=True,
                           sort_close_to=None,
                           find_3dmin=True, minimize_options=dict(),
                           verbose=True, title=''):

    if verbose:
        print(_color_str(Fore.YELLOW, f"--------------\nAnalyse potential for ion {ion}: {title}"))
    if find_3dmin:
        x1, y1, z1 = find_3dmin_potential(trap, voltages, ion, r0,
                                          roi, pseudo, minimize_options, verbose)
    else:
        if verbose:
            print(_color_str(Fore.YELLOW, "Set position to r0"))
        x1, y1, z1 = r0

    v = trap.potential(voltages, x1, y1, z1, ion.mass_amu, pseudo=pseudo)
    E = trap.gradient(voltages, x1, y1, z1, ion.mass_amu, pseudo=pseudo)
    H = trap.hessian(voltages, x1, y1, z1, ion.mass_amu, pseudo=pseudo)

    h, vs, angle = _eig_hessian(H, sort_close_to)
    freqs = curv_to_freq(h, ion=ion)
    shift = field_to_shift(E, ion=ion)

    if verbose:
        with np.printoptions(suppress=True):
            print(_color_str(Fore.YELLOW, 'Gradient [V/m]'))
            print(E)
            print(_color_str(Fore.YELLOW, "Displacement at 1 MHz [um]"))
            print(shift * 1e6)
            print(_color_str(Fore.YELLOW, 'Hessian [V/m2]'))
            print(H)
            # print(curv_to_freq(H, ion=ion) * 1e-6)
            print(_color_str(Fore.YELLOW, "Normal mode frequencies [MHz]"))
            with np.printoptions(formatter={'float': lambda x: f"{x * 1e-6:g}" if x > 0 else _color_str(Fore.RED, f"{x * 1e-6:g}")}):
                print(freqs)
            print(_color_str(Fore.YELLOW, 'Eigenvectors'))
            with np.printoptions(formatter={'float': lambda x: _color_str(Fore.GREEN, f"{x:.3g}") if abs(x) > 0.9 else f"{x:.3g}"}):
                print(vs)
            print(_color_str(Fore.YELLOW, f"Tilt angle of mode 2 ({freqs[2] * 1e-6:.2f}):") + f" {angle:.2f}°")
        print()

    results = dict(
        fun=v,
        x=x1,
        y=y1,
        z=z1,
        fx=freqs[0],
        fy=freqs[1],
        fz=freqs[2],
        fields=E,
        shift=shift,
        hessian=H,
        r1=np.asarray([x1, y1, z1]),
        freqs=freqs,
        eigenvalues=h,
        eigenvectors=vs,
        angle=angle
    )
    return results


def analyse_potential(trap: AbstractTrapModel, voltages: NDArray, ions: Union[Ion, List[Ion]],
                      r0: NDArray[Shape["3"], Any],
                      plot=True, axes=None, title='',
                      roi=(400, 30, 30), pseudo=True, find_3dmin=True, minimize_options=dict(), verbose=True):

    res = mode_solver(trap, voltages, ions, x0, bounding_box=None, minimize_options=dict())
    if not plot:
        return res

    if axes is None:
        fig, axes = plot3d_make_layout(n=1)
    roi = __roi if roi is None else roi

    fig, axes = plot3d_potential(trap, voltages, ion, r0, analyse_results=res, roi=roi, axes=axes, pseudo=pseudo, title=title)

    res['fig'] = fig
    res['axes'] = axes

    return res
