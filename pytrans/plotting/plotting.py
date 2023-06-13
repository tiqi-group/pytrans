#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 03/2023
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

import numpy as np
from typing import Optional
from nptyping import NDArray
from pytrans.typing import Coords1, Roi

from pytrans.analysis.results import AnalysisResults

from pytrans.abstract_model import AbstractTrapModel
from pytrans.ions import Ion

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import make_axes
from matplotlib import patches as mpatches
from matplotlib import transforms


def plot3d_potential(trap: AbstractTrapModel, voltages: NDArray, ion: Ion, r0: Coords1,
                     roi: Roi, axes=None, trapAxis='x', radialAxes=['y', 'z'], pseudo=True, analyse_results: Optional[AnalysisResults] = None, title=''):

    if axes is None:
        fig, axes = plot3d_make_layout(n=1)

    ax_tAxis, ax_rAxis0, ax_rAxis1, ax_im, ax0 = axes
    fig = ax_tAxis.figure

    ax_im.get_shared_x_axes().join(ax_im, ax_rAxis0)
    ax_im.get_shared_y_axes().join(ax_im, ax_rAxis1)

    # x0, y0, z0 = r0
    mapper = {
        'tAxis': (trapAxis == 'x') * 0 + (trapAxis == 'y') * 1 + (trapAxis == 'z') * 2,
        'rAxis0': (radialAxes[0] == 'x') * 0 + (radialAxes[0] == 'y') * 1 + (radialAxes[0] == 'z') * 2,
        'rAxis1': (radialAxes[1] == 'x') * 0 + (radialAxes[1] == 'y') * 1 + (radialAxes[1] == 'z') * 2
    }
    x0 = r0[mapper['tAxis']]
    y0 = r0[mapper['rAxis0']]
    z0 = r0[mapper['rAxis1']]

    _roi = []
    for key in ['tAxis', 'rAxis0', 'rAxis1']:
        lim = roi[mapper[key]]
        lim = (-lim, lim) if isinstance(lim, (int, float)) else lim
        _roi.append(lim)

    lx, ly, lz = _roi
    _tAxis = np.linspace(lx[0], lx[1], 100)
    _rAxis0 = np.linspace(ly[0], ly[1], 100)
    _rAxis1 = np.linspace(lz[0], lz[1], 100)

    _xyz = np.stack([_tAxis, _rAxis0, _rAxis1], axis=0)

    tAxis, rAxis0, rAxis1 = _xyz + np.asarray(r0).reshape((-1, 1))

    def _fun(x, y, z):
        return trap.potential(voltages, x, y, z, ion.mass_amu, pseudo=pseudo)

    fun_args = [0, 0, 0]
    fun_args[mapper['tAxis']], fun_args[mapper['rAxis0']], fun_args[mapper['rAxis1']] = tAxis, y0, z0
    ax_tAxis.plot(tAxis * 1e6, _fun(*fun_args))
    fun_args[mapper['tAxis']], fun_args[mapper['rAxis0']], fun_args[mapper['rAxis1']] = x0, rAxis0, z0
    ax_rAxis0.plot(rAxis0 * 1e6, _fun(*fun_args))
    fun_args[mapper['tAxis']], fun_args[mapper['rAxis0']], fun_args[mapper['rAxis1']] = x0, y0, rAxis1
    ax_rAxis1.plot(_fun(*fun_args), rAxis1 * 1e6)

    RAXIS0, RAXIS1 = np.meshgrid(rAxis0, rAxis1)
    fun_args[mapper['tAxis']], fun_args[mapper['rAxis0']], fun_args[mapper['rAxis1']] = x0, RAXIS0, RAXIS1
    ps = _fun(*fun_args)

    c0 = ax_im.contour(RAXIS0 * 1e6, RAXIS1 * 1e6, ps, 50)
    try:
        # plt.colorbar(c0, ax=ax_im)
        ax_cb, kk = make_axes(ax0, fraction=0.25, aspect=10)
        plt.colorbar(c0, cax=ax_cb, **kk)
        ax_cb.yaxis.set_ticks_position('left')

    except Exception:
        pass

    # # mark the center of the roi (r0)
    # marker_kw = dict(marker='o', color='none', mfc='none', mec='r')
    # v0 = _fun(*r0)
    # ax_x.plot(x0 * 1e6, v0, **marker_kw)
    # ax_y.plot(y0 * 1e6, v0, **marker_kw)
    # ax_z.plot(v0, z0 * 1e6, **marker_kw)
    # ax_im.plot(y0 * 1e6, z0 * 1e6, **marker_kw)

    # mark RF null
    marker_rf = dict(marker='x', color='none', mec='r', mew=2)
    y_rf = getattr(trap, 'y0', 0)
    z_rf = getattr(trap, 'z0', 0)
    r_rf = r0[0], y_rf, z_rf
    v_rf = _fun(*r_rf)
    ax_tAxis.plot(x0 * 1e6, v_rf, **marker_rf)
    ax_rAxis0.plot(y_rf * 1e6, v_rf, **marker_rf)
    ax_rAxis1.plot(v_rf, z_rf * 1e6, **marker_rf)
    ax_im.plot(y_rf * 1e6, z_rf * 1e6, **marker_rf)

    ax_tAxis.set(xlabel=trapAxis + ' [um]')
    ax_rAxis0.set(xlabel=radialAxes[0] + ' [um]')
    ax_rAxis1.set(ylabel=radialAxes[1] + ' [um]')
    ax_im.set(title=title, aspect=1)

    if analyse_results is not None:
        plot3d_radial_modes(analyse_results, axes, mapper=mapper)

    return fig, axes


def plot3d_radial_modes(res: AnalysisResults, axes, mapper):
    if res.mode_solver_results is None:
        tAx1, rAx01, rAx11 = res.x_eq[list(mapper.values())]
        rAx0c, rAx1c = rAx01, rAx11
        f1 = res.fun
    else:
        tAx1, rAx01, rAx11 = res.mode_solver_results.x_eq[0, list(mapper.values())].T
        _, rAx0c, rAx1c = res.x_eq[list(mapper.values())]
        f1 = res.mode_solver_results.trap_pot

    freqs = res.mode_freqs
    vs = res.mode_vectors
    angle = res.mode_angle

    ax_tAxis, ax_rAxis0, ax_rAxis1, ax_im, ax0 = axes
    fig = ax_tAxis.figure

    # mark ion(s) positions
    marker_kw = dict(marker='o', mfc='r', mec='r', ls='')

    ax_tAxis.plot(tAx1 * 1e6, f1, **marker_kw)
    ax_rAxis0.plot(rAx01 * 1e6, f1, **marker_kw)
    ax_rAxis1.plot(f1, rAx11 * 1e6, **marker_kw)
    ax_im.plot(rAx01 * 1e6, rAx11 * 1e6, **marker_kw)

    v1 = vs[1, [mapper['rAxis0'], mapper['rAxis1']]]
    v2 = vs[2, [mapper['rAxis0'], mapper['rAxis1']]]
    f1, f2 = freqs[[1, 2]]
    f0 = np.sqrt(abs(f1 * f2))

    tr = fig.dpi_scale_trans + transforms.ScaledTranslation(rAx0c * 1e6, rAx1c * 1e6, ax_im.transData)

    circle = mpatches.Ellipse((0, 0), f0 / f1, f0 / f2, angle=90 + angle,
                              fill=None, transform=tr, color='C0')
    ax_im.add_patch(circle)

    a1 = mpatches.Arrow(0, 0, *v1 * f0 / f1, width=0.2, transform=tr, color='C0')
    ax_im.add_patch(a1)
    a2 = mpatches.Arrow(0, 0, *v2 * f0 / f2, width=0.2, transform=tr, color='C1')
    ax_im.add_patch(a2)


def plot3d_make_axes(fig, left, right, ratio):
    gs = GridSpec(3, 2, fig,
                  height_ratios=[ratio, 1, 1],
                  width_ratios=[1, ratio],
                  wspace=0.1, hspace=0.15,
                  left=left, right=right,
                  top=0.95, bottom=0.1)

    ax_z = fig.add_subplot(gs[0, 0])
    ax_im = fig.add_subplot(gs[0, 1])
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.axis('off')
    ax_y = fig.add_subplot(gs[1, 1])
    ax_x = fig.add_subplot(gs[2, :])
    return ax_x, ax_y, ax_z, ax_im, ax0


def plot3d_make_layout(n, figsize=(5, 6), d=0.08, squeeze=True):
    k = figsize[0] / figsize[1]
    ratio = (2 * k - 1) / (1 - k)
    fig = plt.figure(figsize=(n * figsize[0], figsize[1]))
    axes = [
        plot3d_make_axes(fig, left=k / n + d / 2, right=(k + 1) / n - d / 2, ratio=ratio)
        for k in range(n)
    ]
    if squeeze:
        axes = axes[0] if len(axes) == 1 else axes
    return fig, axes


# def plot_electrodes(ax, electrode_indices=None, y=None, h=None, d=125, L=120, scale=1):
#     d *= scale
#     L *= scale
#     h = np.ptp(ax.get_ylim()) * 0.06 if h is None else h
#     y = min(ax.get_ylim()) + h if y is None else y

#     electrode_indices = range(1, 11) if electrode_indices is None else electrode_indices

#     for n in electrode_indices:
#         c = (n - 6) * d
#         r = Rectangle(((c - L / 2), y - h / 2), L, h, color='gold', zorder=-99)
#         ax.text(c, y, n - 1)
#         ax.add_patch(r)
#     ax.autoscale_view()


def plot_fields_curvatures(x, r0, r1, fields, freqs, angle, title=''):
    fig, (ax_r, ax_e, ax_c) = plt.subplots(1, 3, figsize=(16, 4))
    ax_r.plot(x, r1 * 1e6, label="x y z".split())
    ax_r.legend()
    ax_r.set_prop_cycle(None)
    ax_r.plot(x, r0 * 1e6, ls="--")
    ax_e.plot(x, fields, label="Ex Ey Ez".split())
    ax_e.legend()
    lf = ax_c.plot(x, freqs * 1e-6, label="x r1 r2".split())
    ax2 = ax_c.twinx()
    ax2.format_coord = _make_format(ax2, ax_c)
    la = ax2.plot(x, angle, 'k--', label="angle")
    lines = lf + la
    labels = [line.get_label() for line in lines]
    ax_c.legend(lines, labels)
    fig.suptitle(title)
    ax_r.set_title('Trajectory')
    ax_e.set_title('Fields')
    ax_c.set_title('Normal modes')
    return fig, (ax_e, ax_c)


def _make_format(current, other):
    # https://stackoverflow.com/a/21585524
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x, y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        x1, y1 = inv.transform(display_coord)
        return f"x: {x:.2f}    freq: {y1:.2f}    angle: {y:.2f}"
    return format_coord
