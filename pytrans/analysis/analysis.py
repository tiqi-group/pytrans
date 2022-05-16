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
from pytrans.plotting import plot3d_potential, plot3d_make_layout, plot_fields_curvatures
from pytrans.conversion import curv_to_freq
from pytrans.ions import Ca40
from pytrans.utils.timer import timer
from pytrans.abstract_model import AbstractTrap

from scipy.optimize import minimize
from matplotlib import patches as mpatches
from matplotlib import transforms

from itertools import permutations
from colorama import init as colorama_init, Fore
from tqdm import tqdm

colorama_init(autoreset=True)

__roi = (400, 30, 30)


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


def analyse_fields_curvatures(trap: AbstractTrap, voltages: ArrayLike, x, y=None, z=None, pseudo=True,
                              find_3dmin=False, minimize_options=dict(), plot=True, title=''):
    y = getattr(trap, 'y0', 0) if y is None else y
    z = getattr(trap, 'z0', 0) if z is None else z
    n_samples = len(x)
    samples = np.arange(n_samples)
    r0 = np.stack(np.broadcast_arrays(x, y, z), axis=1)
    print('--------------\n' + Fore.YELLOW + "Analyse curvatures along trajectory")
    r1 = np.empty((n_samples, 3))
    fields = np.empty((n_samples, 3))
    freqs = np.empty((n_samples, 3))
    angle = np.empty_like(x)
    if voltages.ndim == 1:
        voltages = np.tile(voltages, (n_samples, 1))
    for j, x1 in enumerate(tqdm(samples)):
        sort_close_to = None if j == 0 else freqs[j - 1]
        res = analyse_potential_data(trap, voltages[j], r0[j],
                                     pseudo=pseudo, sort_close_to=sort_close_to,
                                     find_3dmin=find_3dmin, minimize_options=minimize_options,
                                     verbose=False)
        r1[j] = res['r1']
        fields[j] = res['fields']
        freqs[j] = res['freqs']
        angle[j] = res['angle']
    results = dict(
        r0=r0,
        r1=r1,
        fields=fields,
        freqs=freqs,
        angle=angle
    )
    if plot:
        results['fig'], results['axes'] = plot_fields_curvatures(samples, r0, r1, fields, freqs, angle, title)
    return results


def find_3dmin_potential(trap, voltages, r0, roi=None, pseudo=True, minimize_options=dict(), verbose=True):
    roi = __roi if roi is None else roi

    def fun3(xyz):
        return trap.potential(voltages, *xyz, pseudo=pseudo)

    _roi = []
    for lim in roi:
        lim = lim if isinstance(lim, (int, float)) else min(lim)
        _roi.append(lim)

    bounds = [(-r * 1e-6 + x, r * 1e-6 + x) for r, x in zip(_roi, r0)]
    opts = dict(accuracy=1e-6)
    opts.update(minimize_options)
    if verbose:
        res = timer(minimize)(fun3, r0, method='TNC', bounds=bounds, options=opts)
    else:
        res = minimize(fun3, r0, method='TNC', bounds=bounds, options=opts)

    if verbose:
        print(Fore.YELLOW + "Potential mimimum [um]")
        print(res.x * 1e6)
        # print((res.x - r0) * 1e6)
    return res.x


def analyse_potential_data(trap: AbstractTrap, voltages: ArrayLike, r0: ArrayLike,
                           roi=None, pseudo=True, sort_close_to=None,
                           find_3dmin=True, minimize_options=dict(),
                           verbose=True):

    if verbose:
        print('--------------\n' + Fore.YELLOW + "Analyse potential")
    if find_3dmin:
        x1, y1, z1 = find_3dmin_potential(trap, voltages, r0, roi, pseudo, minimize_options, verbose)
    else:
        if verbose:
            print(Fore.YELLOW + "Set position to r0")
        x1, y1, z1 = r0

    r1 = np.asarray([x1, y1, z1])
    v = trap.potential(voltages, *r1, pseudo=pseudo)

    E = trap.gradient(voltages, *r1, pseudo=pseudo)
    H = trap.hessian(voltages, *r1, pseudo=pseudo)

    h, vs, angle = _eig_hessian(H, sort_close_to)

    ion = trap.ion if hasattr(trap, 'ion') else Ca40
    freqs = curv_to_freq(h, ion=ion)

    if verbose:
        with np.printoptions(suppress=True):
            print(Fore.YELLOW + 'Gradient [V/m]')
            print(E)
            # print(Fore.YELLOW + f"Displacement for {ion} [um]")
            # print(field_to_shift(E, ion=ion) * 1e6)
            print(Fore.YELLOW + 'Hessian [V/m2]')
            print(H)
            # print(curv_to_freq(H, ion=ion) * 1e-6)
            print(Fore.YELLOW + f"Normal mode frequencies for {ion} [MHz]")
            with np.printoptions(formatter={'float': lambda x: f"{x * 1e-6:g}" if x > 0 else Fore.RED + f"{x * 1e-6:g}" + Fore.RESET}):
                print(freqs)
            print(Fore.YELLOW + 'Eigenvectors')
            with np.printoptions(formatter={'float': lambda x: Fore.GREEN + f"{x:.3g}" + Fore.RESET if abs(x) > 0.9 else f"{x:.3g}"}):
                print(vs)
            print(f"{Fore.YELLOW}Tilt angle of mode 2 ({freqs[2] * 1e-6:.2f}): {Fore.RESET}{angle:.2f}°")
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
        hessian=H,
        r1=r1,
        freqs=freqs,
        eigenvalues=h,
        eigenvectors=vs,
        angle=angle
    )
    return results


def analyse_potential(trap: AbstractTrap, voltages: ArrayLike, r0: ArrayLike,
                      plot=True, axes=None, roi=None,
                      pseudo=True, find_3dmin=True, minimize_options=dict(), title=''):
    if axes is None:
        fig, axes = plot3d_make_layout(n=1)
    roi = __roi if roi is None else roi

    res = analyse_potential_data(trap, voltages, r0, roi, pseudo=pseudo,
                                 find_3dmin=find_3dmin, minimize_options=minimize_options)
    x1, y1, z1 = res['x'], res['y'], res['z']
    freqs = res['fx'], res['fy'], res['fz']
    f1 = res['fun']
    vs = res['eigenvectors']
    angle = res['angle']

    if plot:
        plot3d_potential(trap, voltages, r0, roi=roi, axes=axes, pseudo=pseudo, title=title)

    ax_x, ax_y, ax_z, ax_im, ax0 = axes
    fig = ax_x.figure

    # _range = np.linspace(-roi[0], roi[0], 50) * 1e-6 / 4
    # curv_x = res['eigenvalues'][0]
    # xx1 = _range + x1
    # ax_x.plot(xx1 * 1e6, 0.5 * curv_x * (xx1 - x1)**2 + f1)

    marker_kw = dict(marker='o', mfc='r', mec='r')

    ax_x.plot(x1 * 1e6, f1, **marker_kw)
    ax_y.plot(y1 * 1e6, f1, **marker_kw)
    ax_z.plot(f1, z1 * 1e6, **marker_kw)
    ax_im.plot(y1 * 1e6, z1 * 1e6, **marker_kw)

    v1 = vs[1:, 1]
    v2 = vs[1:, 2]
    f1, f2 = freqs[1], freqs[2]
    f0 = np.sqrt(abs(f1 * f2))

    tr = fig.dpi_scale_trans + transforms.ScaledTranslation(y1 * 1e6, z1 * 1e6, ax_im.transData)

    circle = mpatches.Ellipse((0, 0), f0 / f1, f0 / f2, angle=90 + angle,
                              fill=None, transform=tr, color='C0')
    ax_im.add_patch(circle)

    a1 = mpatches.Arrow(0, 0, *v1 * f0 / f1, width=0.2, transform=tr, color='C0')
    ax_im.add_patch(a1)
    a2 = mpatches.Arrow(0, 0, *v2 * f0 / f2, width=0.2, transform=tr, color='C1')
    ax_im.add_patch(a2)

    res['fig'] = fig
    res['axes'] = axes

    return res
