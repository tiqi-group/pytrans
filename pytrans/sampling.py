#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 03/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
from scipy.interpolate import interp1d


def resample_waveform(waveform, t, axis=0):
    """
    Resample waveform according to new time samples
    t should be
        - a 1d array and have the same size as waveform along the interpolation axis,
          and bounded between 0 and 1
        - an int, which will generate a linearly spaced 1d array batween 0 and 1

    Parameters
        waveform: original waveform, array-like, shape (samples, n_voltages)
        t: new samples, int, or array-like, shape (new_samples,)
        axis: interpolation axis (default = 0)

    Returns
        iterp_waveform: interpolated waveform
    """
    n = waveform.shape[axis]
    x0 = np.linspace(0, 1, n)
    f = interp1d(x0, waveform, axis=axis)
    if isinstance(t, int):
        t = np.linspace(0, 1, t)
    else:
        assert t.min() >= 0 and t.max() <= 1
    return f(t)
