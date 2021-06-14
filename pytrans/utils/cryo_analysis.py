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
from .cryo_solver import tot_potential_ps, tot_hessian_ps
from pytrans.conversion import curv_to_freq

from .timer import timer

from scipy.optimize import minimize as _minimize
from matplotlib import patches as mpatches
from matplotlib import transforms

roi = (400, 30, 30)


@timer
def minimize(*args, **kwargs):
    return _minimize(*args, **kwargs)


def analyse_pot(vv, r0, electrode_indices, Vrf, Omega_rf, axes=None):
    if axes is None:
        fig, axes = plot3d_make_layout(n=1)

    plot_3dpot(tot_potential_ps, r0, args=(vv, electrode_indices, Vrf, Omega_rf), roi=roi, axes=axes)

    ax_x, ax_y, ax_z, ax_im, ax0 = axes
    fig = ax_x.figure
    f_args = (vv, electrode_indices, Vrf, Omega_rf)

    def fun3(xyz):
        return tot_potential_ps(*xyz, *f_args)

    bounds = [(-r * 1e-6 + x, r * 1e-6 + x) for r, x in zip(roi, r0)]

    res = minimize(fun3, r0, method='TNC', bounds=bounds, options=dict(accuracy=1e-3))

    x1, y1, z1 = res.x
    # print(res)

    v0 = res.fun
    H = tot_hessian_ps(x1, y1, z1, *f_args)

    h, vs = np.linalg.eig(H)
    freqs = curv_to_freq(h) * 1e-6

    with np.printoptions(suppress=True):
        print('Hessian')
        print(curv_to_freq(H) * 1e-6)
        print('Eigenvalues [MHz]')
        print(freqs)
        print('Eigenvectors')
        print(vs)

    _range = np.linspace(-roi[0], roi[0], 50) * 1e-6 / 4
    xx1 = _range + x1
    ax_x.plot(xx1 * 1e6, 0.5 * h[0] * (xx1 - x1)**2 + v0)

#     c = ax3.imshow(vs, cmap='RdBu', vmin=-1, vmax=1)
#     plt.colorbar(c, ax=ax3)

    v1 = vs[1:, 1]
    v2 = vs[1:, 2]
    f1, f2 = freqs[[1, 2]]
    f0 = np.sqrt(f1 * f2)

    angle = np.arccos(v2 @ [1, 0]) * 180 / np.pi
    print(f"Tilt angle of mode 2 ({freqs[2]:.2f}): {angle:.2f}°")

    tr = fig.dpi_scale_trans + transforms.ScaledTranslation(y1 * 1e6, z1 * 1e6, ax_im.transData)

    circle = mpatches.Ellipse((0, 0), f0 / f1, f0 / f2, angle=90 + angle,
                              fill=None, transform=tr, color='C0')
    ax_im.add_patch(circle)

    a1 = mpatches.Arrow(0, 0, *v1 * f0 / f1, width=0.2, transform=tr, color='C0')
    ax_im.add_patch(a1)
    a2 = mpatches.Arrow(0, 0, *v2 * f0 / f2, width=0.2, transform=tr, color='C1')
    ax_im.add_patch(a2)

    return res
