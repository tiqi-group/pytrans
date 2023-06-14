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
from matplotlib.patches import Ellipse, Arrow
from matplotlib import transforms

from mpl_toolkits.mplot3d import Axes3D


def plot_potential(trap: AbstractTrapModel, voltages: NDArray, ion: Ion, r0: Coords1,
                   roi: Roi, axes=None, trap_axis='x', pseudo=True, analyse_results: Optional[AnalysisResults] = None, title=''):

    if axes is None:
        fig, axes = plot_potential_make_layout(n=1)

    ax_x, ax_r0, ax_r1, ax_im, ax0 = axes
    fig = ax_x.figure

    # x0, y0, z0 = r0
    _axes = 'xyz'
    ix = _axes.index(trap_axis)
    mapper = {'trap_x': ix, 'trap_r0': (ix + 1) % 3, 'trap_r1': (ix + 2) % 3}
    mapper_slice = list(mapper.values())

    r0 = np.asarray(r0)
    x0, y0, z0 = r0[mapper_slice]

    _roi = []
    for key in ['trap_x', 'trap_r0', 'trap_r1']:
        lim = roi[mapper[key]]
        lim = (-lim, lim) if isinstance(lim, (int, float)) else lim
        _roi.append(lim)

    lx, ly, lz = _roi
    trap_x = np.linspace(lx[0], lx[1], 61) + x0
    trap_r0 = np.linspace(ly[0], ly[1], 61) + y0
    trap_r1 = np.linspace(lz[0], lz[1], 61) + z0

    def _fun(x, y, z):
        return trap.potential(voltages, x, y, z, ion.mass_amu, pseudo=pseudo)

    fun_args = np.empty((3,), dtype=object)

    fun_args[mapper_slice] = trap_x, y0, z0
    ax_x.plot(trap_x * 1e6, _fun(*fun_args))

    fun_args[mapper_slice] = x0, trap_r0, z0
    ax_r0.plot(trap_r0 * 1e6, _fun(*fun_args))

    fun_args[mapper_slice] = x0, y0, trap_r1
    ax_r1.plot(_fun(*fun_args), trap_r1 * 1e6)

    trap_r0, trap_r1 = np.meshgrid(trap_r0, trap_r1)
    fun_args[mapper_slice] = x0, trap_r0, trap_r1
    ps = _fun(*fun_args)

    c0 = ax_im.contour(trap_r0 * 1e6, trap_r1 * 1e6, ps, 30)
    try:
        # plt.colorbar(c0, ax=ax_im)
        ax_cb, kk = make_axes(ax0, fraction=0.25, aspect=10)
        plt.colorbar(c0, cax=ax_cb, **kk)
        ax_cb.yaxis.set_ticks_position('left')

    except Exception:
        pass

    # mark RF null
    # TODO: This is still targeted at traps with axis x
    marker_rf = dict(marker='x', color='none', mec='r', mew=2)
    y_rf = getattr(trap, 'y0', 0)
    z_rf = getattr(trap, 'z0', 0)
    r_rf = r0[0], y_rf, z_rf
    v_rf = _fun(*r_rf)
    ax_x.plot(x0 * 1e6, v_rf, **marker_rf)
    ax_r0.plot(y_rf * 1e6, v_rf, **marker_rf)
    ax_r1.plot(v_rf, z_rf * 1e6, **marker_rf)
    ax_im.plot(y_rf * 1e6, z_rf * 1e6, **marker_rf)

    ax_x.set(xlabel=_axes[mapper['trap_x']] + ' [um]')
    ax_r0.set(xlabel=_axes[mapper['trap_r0']] + ' [um]')
    ax_r1.set(ylabel=_axes[mapper['trap_r1']] + ' [um]')
    ax_im.set(title=title, aspect=1)

    if analyse_results is not None:
        plot_ion_positions(axes, analyse_results, mapper=mapper)
        plot_mode_vectors(ax_im, analyse_results, mapper=mapper)

    return fig, axes


def plot3d_potential(trap: AbstractTrapModel, voltages: NDArray, ion: Ion, r0: Coords1, roi: Roi,
                     pseudo=True, analyse_results: Optional[AnalysisResults] = None, title=''):

    x0, y0, z0 = r0

    _roi = []
    for lim in roi:
        lim = (-lim, lim) if isinstance(lim, (int, float)) else lim
        _roi.append(lim)

    lx, ly, lz = _roi
    x = np.linspace(lx[0], lx[1], 51) + x0
    y = np.linspace(ly[0], ly[1], 51) + y0
    z = np.linspace(lz[0], lz[1], 51) + z0

    # Create the figure and subplots
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def _fun(x, y, z):
        return trap.potential(voltages, x, y, z, ion.mass_amu, pseudo=pseudo)

    # Plot the equipotential surface
    # TODO: this actualy needs a good 3D contour algorithm, which I couldn't find
    # X, Y = np.meshgrid(x, y)
    # c = _fun(x0, y0 + 5e-6, z0)
    # Z = np.zeros_like(X)
    # for i in range(len(x)):
    #     for j in range(len(y)):
    #         def f(z):
    #             return _fun(X[i, j], Y[i, j], z) - c
    #         try:
    #             Z[i, j] = brentq(f, 0, z.max())
    #         except ValueError:
    #             Z[i, j] = np.nan
    # ax.plot_surface(X, Y, Z, cmap='viridis')

    # Plot the contour slices along the principal axes on the walls
    _kwargs = dict(levels=30, cmap='coolwarm', alpha=0.65)

    X, Y = np.meshgrid(x, y)
    xy_slice = _fun(X, Y, z0)
    ax.contour(x * 1e6, y * 1e6, xy_slice, zdir='z', offset=z.min() * 1e6, **_kwargs)

    Y, Z = np.meshgrid(y, z)
    yz_slice = _fun(x0, Y, Z)
    ax.contour(yz_slice, y * 1e6, z * 1e6, zdir='x', offset=x.min() * 1e6, **_kwargs)

    Z, X = np.meshgrid(z, x)
    xz_slice = _fun(X, y0, Z)
    ax.contour(x * 1e6, xz_slice, z * 1e6, zdir='y', offset=y.max() * 1e6, **_kwargs)

    ax.set(
        xlabel='x [um]',
        ylabel='y [um]',
        zlabel='z [um]',
        xlim=(x.min() * 1e6, x.max() * 1e6),
        ylim=(y.min() * 1e6, y.max() * 1e6),
        zlim=(z.min() * 1e6, z.max() * 1e6),
        title=title,
        aspect='equal'
    )

    if analyse_results is not None:
        plot_ion_positions(ax, analyse_results)

    return fig, ax


def plot_ion_positions(axes, res: AnalysisResults, mapper=None):
    mapper_slice = list(mapper.values()) if mapper is not None else list(range(3))
    if res.mode_solver_results is None:
        x1, r0, r1 = res.x_eq[mapper_slice]
        r0c, r1c = r0, r1
        f1 = res.fun
    else:
        x1, r0, r1 = res.mode_solver_results.x_eq[:, mapper_slice].T
        _, r0c, r1c = res.x_eq[mapper_slice]
        f1 = res.mode_solver_results.trap_pot

    # mark ion(s) positions
    marker_kw = dict(marker='o', mfc='r', mec='r', ls='')
    if isinstance(axes, Axes3D):
        axes.plot(x1 * 1e6, r0 * 1e6, r1 * 1e6, **marker_kw)
    else:
        ax_x, ax_r0, ax_r1, ax_im, ax0 = axes

        ax_x.plot(x1 * 1e6, f1, **marker_kw)
        ax_r0.plot(r0 * 1e6, f1, **marker_kw)
        ax_r1.plot(f1, r1 * 1e6, **marker_kw)
        ax_im.plot(r0 * 1e6, r1 * 1e6, **marker_kw)


def plot_mode_vectors(ax, res: AnalysisResults, mapper):
    r0 = res.x_eq
    mode_freqs = res.mode_freqs
    mode_vectors = res.mode_vectors
    indices = [mapper['trap_r0'], mapper['trap_r1']]
    v1 = mode_vectors[indices[0], indices]
    v2 = mode_vectors[indices[1], indices]
    r0c, r1c = r0[indices]
    f1, f2 = mode_freqs[indices]
    f0 = np.sqrt(abs(f1 * f2))
    angle = np.arctan2(1, v2[0] / v2[1]) * 180 / np.pi
    fig = ax.figure
    tr = fig.dpi_scale_trans + transforms.ScaledTranslation(r0c * 1e6, r1c * 1e6, ax.transData)

    circle = Ellipse((0, 0), f0 / f1, f0 / f2, angle=90 + angle,
                     fill=None, transform=tr, color='C0')
    ax.add_patch(circle)

    arrow_kwargs = dict(width=0.2, transform=tr)
    a1 = Arrow(0, 0, *v1 * f0 / f1, **arrow_kwargs)
    ax.add_patch(a1)
    a2 = Arrow(0, 0, *v2 * f0 / f2, **arrow_kwargs)
    ax.add_patch(a2)


def _plot3d_mode_vectors(ax: Axes3D, res: AnalysisResults):
    r0 = res.x_eq
    mode_freqs = res.mode_freqs
    mode_vectors = res.mode_vectors
    f0 = pow(abs(np.prod(mode_freqs)), 1 / len(mode_freqs))

    widths = np.asarray(ax.get_w_lims()).reshape(3, 2).ptp(axis=1)
    dr = widths / 6 * f0 / mode_freqs
    r0 *= 1e6
    for k in range(3):
        r1 = r0.copy()
        r1 += mode_vectors[:, k] * dr[k]
        r = np.stack([r0, r1], axis=1)
        ax.plot(*r)


def plot_potential_make_axes(fig, left, right, ratio):
    gs = GridSpec(3, 2, fig,
                  height_ratios=[ratio, 1, 1],
                  width_ratios=[1, ratio],
                  wspace=0.1, hspace=0.15,
                  left=left, right=right,
                  top=0.95, bottom=0.1)

    ax_y = fig.add_subplot(gs[1, 1])
    ax_z = fig.add_subplot(gs[0, 0])
    ax_im = fig.add_subplot(gs[0, 1], sharex=ax_y, sharey=ax_z)
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.axis('off')
    ax_x = fig.add_subplot(gs[2, :])
    return ax_x, ax_y, ax_z, ax_im, ax0


def plot_potential_make_layout(n, figsize=(5, 6), d=0.08, squeeze=True):
    k = figsize[0] / figsize[1]
    assert k < 1
    ratio = (2 * k - 1) / (1 - k)
    fig = plt.figure(figsize=(n * figsize[0], figsize[1]))
    axes = [
        plot_potential_make_axes(fig, left=k / n + d / 2, right=(k + 1) / n - d / 2, ratio=ratio)
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
