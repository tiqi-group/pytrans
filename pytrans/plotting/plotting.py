#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 03/2023
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

import numpy as np
from typing import TYPE_CHECKING, Union, List, Optional
from numpy.typing import NDArray
from pytrans.typing import Coords1, RoiSize, Bounds

from .roi import Roi

if TYPE_CHECKING:
    from pytrans.analysis.results import AnalysisResults

from pytrans.abstract_model import AbstractTrapModel
from pytrans.ions import Ion

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import make_axes
from matplotlib.lines import Line2D
from matplotlib import patches as mpc
from matplotlib import transforms

from mpl_toolkits.mplot3d import Axes3D


def plot_potential(
    trap: AbstractTrapModel,
    voltages: NDArray,
    ion: Ion,
    r0: Coords1,
    roi: Union[RoiSize, Bounds],
    axes=None,
    trap_axis="x",
    pseudo=True,
    plot_colorbar=False,
    analyse_results: Optional["AnalysisResults"] = None,
    title="",
):
    if axes is None:
        fig, axes = plot_potential_make_layout(n=1)

    ax_x, ax_r0, ax_r1, ax_im, ax0 = axes
    fig = ax_x.figure

    # x0, y0, z0 = r0
    _axes = "xyz"
    ix = _axes.index(trap_axis)
    mapper = {"trap_x": ix, "trap_r0": (ix + 1) % 3, "trap_r1": (ix + 2) % 3}
    mapper_slice = list(mapper.values())

    roi = Roi(roi, r0)
    x0, y0, z0 = [r0[ix] for ix in mapper_slice]
    lx, ly, lz = [roi.bounds[ix] for ix in mapper_slice]

    trap_x = np.linspace(lx[0], lx[1], 61)
    trap_r0 = np.linspace(ly[0], ly[1], 61)
    trap_r1 = np.linspace(lz[0], lz[1], 61)

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
    if plot_colorbar:
        try:
            # plt.colorbar(c0, ax=ax_im)
            ax_cb, kk = make_axes(ax0, fraction=0.25, aspect=10, location="right")
            plt.colorbar(c0, cax=ax_cb, **kk)
            # ax_cb.locator_params(nbins=1)
            # ax_cb.yaxis.set_ticks_position('left')
            ax_cb.set_yticks([])

        except Exception:
            pass

    # mark r0
    f1 = _fun(x0, y0, z0)
    marker_kw = dict(marker="+", c="k", zorder=89)
    ax_x.scatter(x0 * 1e6, f1, **marker_kw)
    ax_r0.scatter(y0 * 1e6, f1, **marker_kw)
    ax_r1.scatter(f1, z0 * 1e6, **marker_kw)
    ax_im.scatter(y0 * 1e6, z0 * 1e6, **marker_kw)

    # mark RF null
    try:
        plot_rf_null(ax_im, trap.rf_null_coords, mapper)
    except Exception:
        pass

    ax_x.text(0.99, 0.05, _axes[mapper["trap_x"]] + " [um]", ha="right", va="bottom", transform=ax_x.transAxes)
    ax_r0.text(0.99, 0.05, _axes[mapper["trap_r0"]] + " [um]", ha="right", va="bottom", transform=ax_r0.transAxes)
    ax_r1.text(0.05, 0.99, _axes[mapper["trap_r1"]] + " [um]", ha="left", va="top", transform=ax_r1.transAxes)
    # ax_x.set(xlabel=_axes[mapper['trap_x']] + ' [um]')
    # ax_r0.set(xlabel=_axes[mapper['trap_r0']] + ' [um]')
    # ax_r1.set(ylabel=_axes[mapper['trap_r1']] + ' [um]')
    ax_im.set(title=title)

    if analyse_results is not None:
        plot_ion_positions(axes, analyse_results, mapper=mapper)
        plot_mode_vectors(ax_im, analyse_results, mapper=mapper)

    return fig, axes


def plot3d_potential(
    trap: AbstractTrapModel,
    voltages: NDArray,
    ion: Ion,
    r0: Coords1,
    roi: Union[RoiSize, Bounds],
    pseudo=True,
    analyse_results: Optional["AnalysisResults"] = None,
    title="",
):
    x0, y0, z0 = r0

    roi = Roi(roi, r0)
    lx, ly, lz = roi.bounds
    x = np.linspace(lx[0], lx[1], 61)
    y = np.linspace(ly[0], ly[1], 61)
    z = np.linspace(lz[0], lz[1], 61)

    # Create the figure and subplots
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

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
    _kwargs = dict(levels=30, cmap="coolwarm", alpha=0.65, zorder=-1)

    X, Y = np.meshgrid(x, y)
    xy_slice = _fun(X, Y, z0)
    ax.contour(x * 1e6, y * 1e6, xy_slice, zdir="z", offset=z.min() * 1e6, **_kwargs)

    Y, Z = np.meshgrid(y, z)
    yz_slice = _fun(x0, Y, Z)
    ax.contour(yz_slice, y * 1e6, z * 1e6, zdir="x", offset=x.min() * 1e6, **_kwargs)

    Z, X = np.meshgrid(z, x)
    xz_slice = _fun(X, y0, Z)
    ax.contour(x * 1e6, xz_slice, z * 1e6, zdir="y", offset=y.max() * 1e6, **_kwargs)

    ax.set(
        xlabel="x [um]",
        ylabel="y [um]",
        zlabel="z [um]",
        xlim=(x.min() * 1e6, x.max() * 1e6),
        ylim=(y.min() * 1e6, y.max() * 1e6),
        zlim=(z.min() * 1e6, z.max() * 1e6),
        title=title,
        aspect="equal",
    )

    if analyse_results is not None:
        plot_ion_positions(ax, analyse_results)

    return fig, ax


def plot_ion_positions(axes, res: "AnalysisResults", mapper=None):
    mapper_slice = list(mapper.values()) if mapper is not None else list(range(3))
    if res.mode_solver_results is None:
        x1, r0, r1 = res.x_eq[mapper_slice]
        r0c, r1c = r0, r1
        f1 = res.fun
        ions = [res.ion]
    else:
        mres = res.mode_solver_results
        x1, r0, r1 = mres.x_eq[:, mapper_slice].T
        _, r0c, r1c = res.x_eq[mapper_slice]
        f1 = mres.trap_pot
        ions = mres.ions

    colors = [_get_ion_color(ion) for ion in ions]
    # m_min = min([ion.mass_amu for ion in ions])
    # sizes = [(ion.mass_amu / m_min) * plt.rcParams['lines.markersize'] ** 2 for ion in ions]
    marker_kw = dict(c=colors, zorder=99)

    # mark ion(s) positions
    if isinstance(axes, Axes3D):
        axes.scatter(x1 * 1e6, r0 * 1e6, r1 * 1e6, **marker_kw)
        _add_ions_legend(axes, ions)
    else:
        ax_x, ax_r0, ax_r1, ax_im, ax0 = axes

        ax_x.scatter(x1 * 1e6, f1, **marker_kw)
        ax_r0.scatter(r0 * 1e6, f1, **marker_kw)
        ax_r1.scatter(f1, r1 * 1e6, **marker_kw)
        ax_im.scatter(r0 * 1e6, r1 * 1e6, **marker_kw)
        _add_ions_legend(ax_x, ions, loc="upper left")


def plot_mode_vectors(ax, res: "AnalysisResults", mapper):
    r0 = res.x_eq
    mode_freqs = res.mode_freqs
    mode_vectors = res.mode_vectors
    indices = [mapper["trap_r0"], mapper["trap_r1"]]
    r0c, r1c = r0[indices]
    fig = ax.figure
    tr = fig.dpi_scale_trans + transforms.ScaledTranslation(r0c * 1e6, r1c * 1e6, ax.transData)

    arrow_kwargs = dict(width=0.2, transform=tr)
    f0 = pow(abs(np.prod(mode_freqs)), 1 / len(mode_freqs))
    f_scale = f0 / mode_freqs
    # f_scale = f_scale / f_scale.max()
    for j, (f, v) in enumerate(zip(f_scale, mode_vectors)):
        v = v[indices]
        a1 = mpc.Arrow(0, 0, *v * f, color=f"C{j}", **arrow_kwargs, label=f"{mode_freqs[j]*1e-6:.2f} MHz")
        ax.add_patch(a1)
    ax.legend()


def plot_rf_null(ax, rf_null_coords, mapper):
    marker_rf = dict(marker="x", color="none", mec="gray", mew=2)
    line_rf = dict(color="gray", lw=1, ls="--")
    y_rf, z_rf = np.asarray(rf_null_coords)[[mapper["trap_r0"], mapper["trap_r1"]]]
    if y_rf is None or y_rf == np.nan:
        ax.axhline(z_rf * 1e6, **line_rf)
    elif z_rf is None or z_rf == np.nan:
        ax.axvline(y_rf * 1e6, **line_rf)
    else:
        ax.plot(y_rf * 1e6, z_rf * 1e6, **marker_rf)


def _plot3d_mode_vectors(ax: Axes3D, res: "AnalysisResults"):
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


def plot_potential_make_axes(fig, subplots_ratio):
    gs = GridSpec(3, 2, fig, height_ratios=[subplots_ratio, 1, 1], width_ratios=[1, subplots_ratio])

    ax_x = fig.add_subplot(gs[2, :])
    ax_y = fig.add_subplot(gs[1, 1])
    ax_z = fig.add_subplot(gs[0, 0])
    ax_im = fig.add_subplot(gs[0, 1], sharex=ax_y, sharey=ax_z)
    ax0 = fig.add_subplot(gs[1, 0])
    for label in ax_im.get_xticklabels():
        label.set_visible(False)
    for label in ax_im.get_yticklabels():
        label.set_visible(False)
    ax_x.set_yticks([])
    ax_y.set_yticks([])
    ax_z.set_xticks([])
    ax_im.set_aspect(1, adjustable="box")
    ax0.axis("off")
    return ax_x, ax_y, ax_z, ax_im, ax0


def plot_potential_make_layout(n, fig_width=None, fig_height=None, subplots_ratio=4, squeeze=True):
    fig_aspect = 4 / 4.8  # width / height
    if fig_width is not None:
        fig_height = fig_width / fig_aspect
    else:
        fig_height = plt.rcParams["figure.figsize"][1] if fig_height is None else fig_height
        fig_width = fig_height * fig_aspect
    wspace = 0.05
    fig = plt.figure(figsize=((n + wspace * (n - 1)) * fig_width, fig_height), layout="compressed")
    subfigures = fig.subfigures(1, n, hspace=0, wspace=wspace, squeeze=False).ravel()
    axes = [plot_potential_make_axes(sfig, subplots_ratio) for sfig in subfigures]
    if squeeze:
        axes = axes[0] if len(axes) == 1 else axes
    return fig, axes


def plot_fields_curvatures(x, r0, r1, fields, freqs, angle, title=""):
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
    la = ax2.plot(x, angle, "k--", label="angle")
    lines = lf + la
    labels = [line.get_label() for line in lines]
    ax_c.legend(lines, labels)
    fig.suptitle(title)
    ax_r.set_title("Trajectory")
    ax_e.set_title("Fields")
    ax_c.set_title("Normal modes")
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


_ion_colors = {"Ca40": "tab:red", "Be9": "tab:blue", "Mg24": "tab:cyan", "Ba138": "tab:purple", "Yb171": "tab:gray"}


def _get_ion_color(ion: Ion):
    if hasattr(ion, "color"):
        return ion.color
    else:
        return _ion_colors.get(str(ion), "black")


def _add_ions_legend(ax, ions: List[Ion], **kwargs):
    handles = []
    for ion in set(ions):
        color = _get_ion_color(ion)
        handles.append(Line2D([0], [0], color=color, marker="o", ls="", label=str(ion)))
    ax.legend(handles=handles, **kwargs)
