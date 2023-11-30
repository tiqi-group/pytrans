#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
import scipy.stats as sstat

from .ions import Ion
from .conversion import freq_to_curv


def gaussian_1d(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / 2 / sigma**2)


def quadratic_potential_1d(x, x0, freq, offset, ion: Ion):
    curv = freq_to_curv(freq, ion=ion)
    return 0.5 * curv * (x - x0) ** 2 + offset


def erfspace(a, b, npts, erf_scaling=2.5):
    """Linspace replacement, producing an error function curve
    man, there's scipy.special.erf for that!
    # TODO: rewrite this
    """
    slope = b - a
    erf_y = sstat.norm.cdf(np.linspace(-erf_scaling, erf_scaling, npts))
    erf_y_slope = erf_y[-1] - erf_y[0]
    vout_zc = erf_y * slope / erf_y_slope  # scale slope
    return vout_zc + a - vout_zc[0]  # shift range


def sinsquared(a, b, npts):
    t = np.linspace(0, 1, npts)
    return a + (b - a) * np.sin(np.pi / 2 * t) ** 2


def zpspace(a, b, npts, k=3, gap=1.5, gap2=None):
    """Linspace replacement, producing a zero-pole curve with adjustable width + smoothness
    Test in linspace_fn_test.org
    """
    if gap2 is None:
        gap2 = gap
    w0 = np.exp(-k)
    w1 = np.exp(k)
    w = np.exp(np.linspace(-gap * k, gap2 * k, npts))
    y = np.log(np.abs((w - 1j * w0) / (w - 1j * w1)))
    return a + (b - a) * (y - y.min()) / (y.max() - y.min())


# Linspace replacement, producing a line with 2 identical points at
# the start and the end looking like a _/-
# def rampspace(a, b, npts, pad=1):
#     assert npts-2*pad >= 2, "Too few points requested for rampspace"
#     return np.hstack([np.repeat(a, pad),
#                       np.linspace(a, b, npts - 2*pad), np.repeat(b, pad)])


def vlinspace(start_vec, end_vec, npts, lin_fn=np.linspace):
    """Linspace on column vectors specifying the starts and ends"""
    assert start_vec.shape == end_vec.shape and start_vec.ndim == 1, "Inputs don't have the same length"
    return np.stack([lin_fn(sv, ev, npts) for sv, ev in zip(start_vec, end_vec)], axis=1)
