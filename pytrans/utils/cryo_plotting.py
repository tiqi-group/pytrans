import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import make_axes
from matplotlib.patches import Rectangle


def plot_3dpot(fun, r0, args=tuple(), roi=(600, 50, 50), axes=None):

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

    x, y, z = _xyz + r0.reshape((-1, 1))

    def _fun(x, y, z):
        return fun(x, y, z, *args)

    v0 = _fun(*r0)

    ax_x.plot(x * 1e6, _fun(x, y0, z0))
    ax_x.plot(x0 * 1e6, v0, 'xr')

    ax_y.plot(y * 1e6, _fun(x0, y, z0))
    ax_y.plot(y0 * 1e6, v0, 'xr')

    ax_z.plot(_fun(x0, y0, z), z * 1e6)
    ax_z.plot(v0, z0 * 1e6, 'xr')
    #     ax_z.plot(z*1e6, _fun(x0, y0, z))
    #     ax_z.plot(z0*1e6, v0, 'xr')

    Y, Z = np.meshgrid(y, z)
    ps = _fun(x0, Y, Z)

    c0 = ax_im.contour(Y * 1e6, Z * 1e6, ps, 50)
    try:
        #         plt.colorbar(c0, ax=ax_im)
        ax_cb, kk = make_axes(ax0, fraction=0.25, aspect=10)
        plt.colorbar(c0, cax=ax_cb, **kk)
        ax_cb.yaxis.set_ticks_position('left')

    except Exception:
        pass

    ax_im.plot(y0 * 1e6, z0 * 1e6, 'or')
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


def plot3d_make_layout(n, figsize=(7, 8), d=0.04, squeeze=True):
    fig = plt.figure(figsize=(n * figsize[0], figsize[1]))
    axes = [
        plot3d_make_axes(fig, left=k / n + d / 2, right=(k + 1) / n - d / 2)
        for k in range(n)
    ]
    if squeeze:
        axes = axes[0] if len(axes) == 1 else axes
    return fig, axes


def plot_electrodes(ax, y=None, h=None, d=125, L=120, scale=1):
    d *= scale
    L *= scale
    h = np.ptp(ax.get_ylim()) * 0.06 if h is None else h
    y = min(ax.get_ylim()) + h if y is None else y

    for n in range(1, 11):
        c = (n - 6) * d
        r = Rectangle(((c - L / 2), y - h / 2), L, h, color='gold', zorder=-99)
        ax.text(c, y, n)
        ax.add_patch(r)
    ax.autoscale_view()
