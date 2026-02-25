#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 03/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
import cvxpy as cx
import scipy.signal as sg
from scipy.linalg import convolution_matrix
from functools import partial

from numpy.typing import NDArray


def pad_waveform(waveform, pad_after, pad_before=0):
    """Extends a waveform (2d numpy array) along the time axis"""
    pad_width = [(pad_before, pad_after)] + [(0, 0)] * (waveform.ndim - 1)
    return np.pad(waveform, pad_width, mode="edge")


def _rename(wname):
    return wname + "_filt"


class TrapFilterTransform:
    def __init__(self, system: sg.dlti):
        assert isinstance(system, sg.dlti)
        self.system = system.to_tf()

        self.zi = sg.lfilter_zi(self.system.num, self.system.den)
        self.impulse_response = np.squeeze(self.system.impulse()[1])

    def transform(self, waveform, pad_after=0):
        if isinstance(waveform, np.ndarray):
            return self._lfilter_waveform_numpy(waveform, pad_after)
        elif isinstance(waveform, cx.Variable):
            return self._lfilter_waveform_cvxpy(waveform, pad_after)
        else:
            raise ValueError(f"Invalid waveform object of type {type(waveform)}")

    def _lfilter_waveform_cvxpy(self, waveform: cx.Variable, pad_after: int) -> cx.Variable:
        pad_before = len(self.impulse_response) - 1
        n, w = waveform.shape
        first_sample = cx.reshape(waveform[0], (1, w))
        last_sample = cx.reshape(waveform[-1], (1, w))
        # stacking copies of the same variable is equivalent to constraining them to be equal
        w0 = cx.vstack([first_sample] * pad_before + [waveform] + [last_sample] * pad_after)
        assert w0.shape == (pad_before + n + pad_after, w)
        M = convolution_matrix(self.impulse_response, w0.shape[0], mode="valid")
        w_filt = M @ w0
        w_filt.name = partial(_rename, waveform.name())
        return w_filt

    def _lfilter_waveform_numpy(self, waveform: NDArray, pad_after: int) -> NDArray:
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
        if pad_after > 0:
            waveform = pad_waveform(waveform, pad_after)
        filtered_waveform, zf = sg.lfilter(
            self.system.num, self.system.den, waveform, axis=0, zi=np.outer(self.zi, waveform[0])
        )
        return filtered_waveform
