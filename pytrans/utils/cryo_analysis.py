#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np

from .cryo_plotting import plot_3dpot, plot3d_make_layout
from .cryo_solver import tot_potential_ps, tot_gradient_ps, tot_hessian_ps
from pytrans.conversion import curv_to_freq as _curv_to_freq, field_to_shift as _field_to_shift
from pytrans.constants import ion_masses, elementary_charge
from .timer import timer

from scipy.optimize import minimize as _minimize
from matplotlib import patches as mpatches
from matplotlib import transforms

from colorama import init as colorama_init, Fore

colorama_init(autoreset=True)

__roi = (400, 30, 30)

mass = ion_masses["Ca"]
charge = elementary_charge


def field_to_shift(e):
    return _field_to_shift(e, mass, charge)


def curv_to_freq(c):
    return _curv_to_freq(c, mass, charge)


@timer
def minimize(*args, **kwargs):
    return _minimize(*args, **kwargs)


def analyse_hessian(H):
    h, vs = np.linalg.eig(H)
    ix = np.argsort(abs(h))
    h = h[ix]
    vs = vs[:, ix]
    # angle = np.arccos(vs[1, 1]) * 180 / np.pi
    angle = np.arctan(vs[2, 2] / vs[1, 2]) * 180 / np.pi
    return h, vs, angle


def analyse_pot(vv, r0, electrode_indices, Vrf, Omega_rf, axes=None, roi=None, find_3dmin=True):
    if axes is None:
        fig, axes = plot3d_make_layout(n=1)
    roi = __roi if roi is None else roi

    res = analyse_pot_data(vv, r0, electrode_indices, Vrf, Omega_rf, roi, find_3dmin)
    x1, y1, z1 = res['x'], res['y'], res['z']
    freqs = res['fx'], res['fy'], res['fz']
    f1 = res['fun']
    vs = res['eigenvectors']
    curv_x = res['eigenvalues'][0]
    angle = res['angle']

    plot_3dpot(tot_potential_ps, r0, args=(vv, electrode_indices, Vrf, Omega_rf), roi=roi, axes=axes)

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


def analyse_pot_data(vv, r0, electrode_indices, Vrf, Omega_rf, roi=None, find_3dmin=True):
    roi = __roi if roi is None else roi

    f_args = (vv, electrode_indices, Vrf, Omega_rf)

    def fun3(xyz):
        return tot_potential_ps(*xyz, *f_args)

    print('--------------\n' + Fore.YELLOW + "Analyse potential")
    if find_3dmin:
        _roi = []
        for lim in roi:
            lim = lim if isinstance(lim, (int, float)) else min(lim)
            _roi.append(lim)

        bounds = [(-r * 1e-6 + x, r * 1e-6 + x) for r, x in zip(_roi, r0)]

        res = minimize(fun3, r0, method='TNC', bounds=bounds, options=dict(accuracy=1e-2))

        print(Fore.YELLOW + "Offset from r0 [um]")
        print((res.x - r0) * 1e6)
        x1, y1, z1 = res.x
        v = res.fun
    else:
        print(Fore.YELLOW + "Set position to r0")
        x1, y1, z1 = r0
        v = fun3(r0)

    E = tot_gradient_ps(x1, y1, z1, *f_args)
    H = tot_hessian_ps(x1, y1, z1, *f_args)

    h, vs, angle = analyse_hessian(H)
    freqs = curv_to_freq(h) * 1e-6

    with np.printoptions(suppress=True):
        print(Fore.YELLOW + 'Gradient')
        print(E)
        print(field_to_shift(E) * 1e6)
        print(Fore.YELLOW + 'Hessian')
        print(H)
        print(curv_to_freq(H) * 1e-6)
        print(Fore.YELLOW + 'Eigenvalues [MHz]')
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
        eigenvalues=h,
        eigenvectors=vs,
        angle=angle
    )
    return results
