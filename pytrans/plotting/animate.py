#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_waveform(trap, waveform, x, y, z, **kwargs):
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))

    nt, nv = waveform.shape
    potentials = trap.potential(waveform, x, y, z)
    print(potentials.shape)
    print(x.shape)

    def find_ylim(a, r=0.05):
        _min = np.min(a)
        _max = np.max(a)
        ptp = _max - _min
        return _min - r * ptp, _max + r * ptp

    ln, = ax.plot(x * 1e6, [0] * len(x),)
    lnv, = ax1.plot(np.arange(nv), [0] * nv)

    def init():
        ax.set_ylim(find_ylim(potentials))
        ax1.set_ylim(find_ylim(waveform))
        return ln, lnv

    def update(j):
        ln.set_ydata(potentials[j])
        lnv.set_ydata(waveform[j])
        return ln, lnv

    kw = dict(blit=True, interval=20, repeat_delay=2000)
    kw.update(kwargs)

    ani = FuncAnimation(fig, update, frames=range(len(waveform)),
                        init_func=init, **kw)
    return ani
