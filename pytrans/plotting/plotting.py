import numpy as np
from numpy.typing import ArrayLike
from pytrans.abstract_model import AbstractTrap

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import make_axes
from matplotlib.patches import Rectangle


def plot3d_potential(trap: AbstractTrap, voltages: ArrayLike, r0: ArrayLike,
                     roi=(600, 50, 50), axes=None):

    if axes is None:
        fig, axes = plot3d_make_layout(n=1)

    ax_x, ax_y, ax_z, ax_im, ax0 = axes
    fig = ax_x.figure

    ax_im.get_shared_x_axes().join(ax_im, ax_y)
    ax_im.get_shared_y_axes().join(ax_im, ax_z)

    x0, y0, z0 = r0

    _roi = []
    for lim in roi:
        lim = (-lim, lim) if isinstance(lim, (int, float)) else lim
        _roi.append(lim)

    lx, ly, lz = _roi
    _x = np.linspace(*lx, 100) * 1e-6
    _y = np.linspace(*ly, 100) * 1e-6
    _z = np.linspace(*lz, 100) * 1e-6

    _xyz = np.stack([_x, _y, _z], axis=0)

    x, y, z = _xyz + np.asarray(r0).reshape((-1, 1))

    def _fun(x, y, z):
<<<<<<< Updated upstream
        return trap.potential(voltages, x, y, z)
=======
        return trap.potential(voltages, x, y, z, pseudo=pseudo)
>>>>>>> Stashed changes

    ax_x.plot(x * 1e6, _fun(x, y0, z0))
    ax_y.plot(y * 1e6, _fun(x0, y, z0))
    ax_z.plot(_fun(x0, y0, z), z * 1e6)

    Y, Z = np.meshgrid(y, z)
    ps = _fun(x0, Y, Z)

    c0 = ax_im.contour(Y * 1e6, Z * 1e6, ps, 50)
    try:
        # plt.colorbar(c0, ax=ax_im)
        ax_cb, kk = make_axes(ax0, fraction=0.25, aspect=10)
        plt.colorbar(c0, cax=ax_cb, **kk)
        ax_cb.yaxis.set_ticks_position('left')

    except Exception:
        pass

    # mark the center of the roi (r0)
    marker_kw = dict(marker='o', color='none', mfc='none', mec='r')
    v0 = _fun(*r0)
    ax_x.plot(x0 * 1e6, v0, **marker_kw)
    ax_y.plot(y0 * 1e6, v0, **marker_kw)
    ax_z.plot(v0, z0 * 1e6, **marker_kw)
    ax_im.plot(y0 * 1e6, z0 * 1e6, **marker_kw)

    # mark RF null
    marker_rf = dict(marker='x', color='none', mec='r', mew=2)
    y_rf = getattr(trap, 'y0', 0)
    z_rf = getattr(trap, 'z0', 0)
    r_rf = r0[0], y_rf, z_rf
    v_rf = _fun(*r_rf)
    ax_x.plot(x0 * 1e6, v_rf, **marker_rf)
    ax_y.plot(y_rf * 1e6, v_rf, **marker_rf)
    ax_z.plot(v_rf, z_rf * 1e6, **marker_rf)
    ax_im.plot(y_rf * 1e6, z_rf * 1e6, **marker_rf)

    ax_x.set(xlabel='x [um]')
    ax_y.set(xlabel='y [um]')
    ax_z.set(ylabel='z [um]')
    #     ax_im.set(
    #         xlabel='y [um]',
    #         ylabel='z [um]'
    #     )

    plot_electrodes(ax_x)

    return fig, axes


def plot3d_make_axes(fig, left, right):
    gs = GridSpec(3,
                  2,
                  fig,
                  height_ratios=[3, 1, 1],
                  width_ratios=[1, 3],
                  left=left,
                  right=right)

    ax_z = fig.add_subplot(gs[0, 0])
    ax_im = fig.add_subplot(gs[0, 1])
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.axis('off')
    ax_y = fig.add_subplot(gs[1, 1])
    ax_x = fig.add_subplot(gs[2, :])
    return ax_x, ax_y, ax_z, ax_im, ax0


def plot3d_make_layout(n, figsize=(8, 7), d=0.04, squeeze=True):
    fig = plt.figure(figsize=(n * figsize[0], figsize[1]))
    axes = [
        plot3d_make_axes(fig, left=k / n + d / 2, right=(k + 1) / n - d / 2)
        for k in range(n)
    ]
    if squeeze:
        axes = axes[0] if len(axes) == 1 else axes
    return fig, axes


def plot_electrodes(ax, electrode_indices=None, y=None, h=None, d=125, L=120, scale=1):
    d *= scale
    L *= scale
    h = np.ptp(ax.get_ylim()) * 0.06 if h is None else h
    y = min(ax.get_ylim()) + h if y is None else y

    electrode_indices = range(1, 11) if electrode_indices is None else electrode_indices

    for n in electrode_indices:
        c = (n - 6) * d
        r = Rectangle(((c - L / 2), y - h / 2), L, h, color='gold', zorder=-99)
        ax.text(c, y, n - 1)
        ax.add_patch(r)
    ax.autoscale_view()
<<<<<<< Updated upstream
=======


def plot_curvatures(x, modes, angle, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    lf = ax.plot(x * 1e6, modes * 1e-6, label="x r1 r2".split())
    ax2 = ax.twinx()
    ax2.format_coord = _make_format(ax2, ax)
    la = ax2.plot(x * 1e6, angle, 'k--', label="angle")
    lines = lf + la
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)


def plot_fields_curvatures(x, fields, modes, angle):
    fig, (ax_e, ax_c) = plt.subplots(1, 2, figsize=(12, 4))
    ax_e.plot(x * 1e6, fields, label="Ex Ey Ez".split())
    ax_e.legend()
    plot_curvatures(x, modes, angle, ax=ax_c)
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
>>>>>>> Stashed changes
