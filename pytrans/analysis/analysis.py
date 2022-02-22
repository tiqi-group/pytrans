#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np
from numpy.typing import ArrayLike
from pytrans.plotting import plot3d_potential, plot3d_make_layout, plot_curvatures
from pytrans.conversion import curv_to_freq, field_to_shift
from pytrans.ions import Ca40
from pytrans.utils.timer import timer
from pytrans.abstract_model import AbstractTrap

from scipy.optimize import minimize
from matplotlib import patches as mpatches
from matplotlib import transforms

from colorama import init as colorama_init, Fore
from tqdm import tqdm

colorama_init(autoreset=True)

__roi = (400, 30, 30)


minimize = timer(minimize)


def _eig_hessian(H):
    h, vs = np.linalg.eig(H)
    ix = np.argsort(abs(h))
    h = h[ix]
    vs = vs[:, ix]
    angle = np.arctan2(1, vs[1, 2] / vs[2, 2]) * 180 / np.pi
    return h, vs, angle


def analyse_curvatures(trap: AbstractTrap, voltages: ArrayLike, x, y=None, z=None, plot=True, ax=None):
    y = getattr(trap, 'y0', 0) if y is None else y
    z = getattr(trap, 'z0', 0) if z is None else z
    print('--------------\n' + Fore.YELLOW + "Analyse curvatures along trajectory")
    modes = np.empty(x.shape + (3,))
    angle = np.empty_like(x)
    for j, x1 in enumerate(tqdm(x)):
        H = trap.hessian(voltages[j], x1, y, z)
        h, _, angle[j] = _eig_hessian(H)
        modes[j] = curv_to_freq(h, ion=trap.ion)
    if plot:
        plot_curvatures(modes, angle, ax)
    return modes, angle


def analyse_potential_data(trap: AbstractTrap, voltages: ArrayLike, r0: ArrayLike,
                           roi=None, find_3dmin=True, minimize_options=dict()):
    roi = __roi if roi is None else roi

    def fun3(xyz):
        return trap.potential(voltages, *xyz)

    print('--------------\n' + Fore.YELLOW + "Analyse potential")
    if find_3dmin:
        _roi = []
        for lim in roi:
            lim = lim if isinstance(lim, (int, float)) else min(lim)
            _roi.append(lim)

        bounds = [(-r * 1e-6 + x, r * 1e-6 + x) for r, x in zip(_roi, r0)]
        opts = dict(accuracy=1e-6)
        opts.update(minimize_options)
        res = minimize(fun3, r0, method='TNC', bounds=bounds, options=opts)

        print(Fore.YELLOW + "Offset from r0 [um]")
        print((res.x - r0) * 1e6)
        x1, y1, z1 = res.x
        v = res.fun
    else:
        print(Fore.YELLOW + "Set position to r0")
        x1, y1, z1 = r0
        v = fun3(r0)

    E = trap.gradient(voltages, x1, y1, z1)
    H = trap.hessian(voltages, x1, y1, z1)

    h, vs, angle = _eig_hessian(H)

    ion = trap.ion if hasattr(trap, 'ion') else Ca40
    freqs = curv_to_freq(h, ion=ion) * 1e-6

    with np.printoptions(suppress=True):
        print(Fore.YELLOW + 'Gradient [V/m]')
        print(E)
        print(Fore.YELLOW + f"Displacement for {ion} [um]")
        print(field_to_shift(E, ion=ion) * 1e6)
        print(Fore.YELLOW + 'Hessian [V/m2]')
        print(H)
        # print(curv_to_freq(H, ion=ion) * 1e-6)
        print(Fore.YELLOW + f"Normal mode frequencies for {ion} [MHz]")
        with np.printoptions(formatter={'float': lambda x: f"{x:g}" if x > 0 else Fore.RED + f"{x:g}" + Fore.RESET}):
            print(freqs)
        print(Fore.YELLOW + 'Eigenvectors')
        with np.printoptions(formatter={'float': lambda x: Fore.GREEN + f"{x:.3g}" + Fore.RESET if abs(x) > 0.9 else f"{x:.3g}"}):
            print(vs)
        print(f"{Fore.YELLOW}Tilt angle of mode 2 ({freqs[2]:.2f}): {Fore.RESET}{angle:.2f}°")
    print()

    results = dict(
        fun=v,
        x=x1,
        y=y1,
        z=z1,
        fx=freqs[0],
        fy=freqs[1],
        fz=freqs[2],
        hessian=H,
        eigenvalues=h,
        eigenvectors=vs,
        angle=angle
    )
    return results


def analyse_potential(trap: AbstractTrap, voltages: ArrayLike, r0: ArrayLike, plot=True,
                      axes=None, roi=None, find_3dmin=True, minimize_options=dict()):
    if axes is None:
        fig, axes = plot3d_make_layout(n=1)
    roi = __roi if roi is None else roi

    res = analyse_potential_data(trap, voltages, r0, roi, find_3dmin, minimize_options)
    x1, y1, z1 = res['x'], res['y'], res['z']
    freqs = res['fx'], res['fy'], res['fz']
    f1 = res['fun']
    vs = res['eigenvectors']
    curv_x = res['eigenvalues'][0]
    angle = res['angle']

    if plot:
        plot3d_potential(trap, voltages, r0, roi=roi, axes=axes)

    ax_x, ax_y, ax_z, ax_im, ax0 = axes
    fig = ax_x.figure

    _range = np.linspace(-roi[0], roi[0], 50) * 1e-6 / 4
    xx1 = _range + x1
    ax_x.plot(xx1 * 1e6, 0.5 * curv_x * (xx1 - x1)**2 + f1)

    marker_kw = dict(marker='o', mfc='r', mec='r')

    ax_x.plot(x1 * 1e6, f1, **marker_kw)
    ax_y.plot(y1 * 1e6, f1, **marker_kw)
    ax_z.plot(f1, z1 * 1e6, **marker_kw)
    ax_im.plot(y1 * 1e6, z1 * 1e6, **marker_kw)

    v1 = vs[1:, 1]
    v2 = vs[1:, 2]
    f1, f2 = freqs[1], freqs[2]
    f0 = np.sqrt(f1 * f2)

    tr = fig.dpi_scale_trans + transforms.ScaledTranslation(y1 * 1e6, z1 * 1e6, ax_im.transData)

    circle = mpatches.Ellipse((0, 0), f0 / f1, f0 / f2, angle=90 + angle,
                              fill=None, transform=tr, color='C0')
    ax_im.add_patch(circle)

    a1 = mpatches.Arrow(0, 0, *v1 * f0 / f1, width=0.2, transform=tr, color='C0')
    ax_im.add_patch(a1)
    a2 = mpatches.Arrow(0, 0, *v2 * f0 / f2, width=0.2, transform=tr, color='C1')
    ax_im.add_patch(a2)

    return res
