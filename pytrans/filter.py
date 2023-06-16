#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 03/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
import scipy.signal as sg


def pad_waveform(waveform, pad_after, pad_before=0):
    """Extends a waveform along the time axis
    """
    pad_width = [(pad_before, pad_after)] + [(0, 0)] * (waveform.ndim - 1)
    return np.pad(waveform, pad_width, mode='edge')


def lfilter_waveform(b, a, waveform, pad_after=None):
    """Filters a waveform using a digital filter
    in the transfer function representation
                        -1              -M
            b[0] + b[1]z  + ... + b[M] z
    Y(z) = -------------------------------- X(z)
                        -1              -N
            a[0] + a[1]z  + ... + a[N] z

    Parameters:
        b: array_like
        a: array_like
        waveform: array_like, shape (n_samples, n_electrodes)

    Returns
        filtered_waveform
    """
    zi = sg.lfilter_zi(b, a)
    if pad_after is not None:
        waveform = pad_waveform(waveform, pad_after)
    filtered_waveform, zf = sg.lfilter(b, a, waveform, axis=0, zi=np.outer(zi, waveform[0]))
    return filtered_waveform
