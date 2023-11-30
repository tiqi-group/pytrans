#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

"""
Module docstring
"""

import numpy as np
from .ions import Ion


def curv_to_freq(curv, mass=None, charge=None, ion: Ion = None):
    """
    Secular frequency for the given curvature and ion

    Parameters
        curv: potential curvature [V/m^2]
        mass: mass of the ion [kg]
        charge: charge of the ion [C]
        ion (optional): ion species class specifying mass and charge

    Returns
        freq: secular frequency [Hz]
    """
    if ion is not None:
        mass = ion.mass
        charge = ion.charge
    C = (2 * np.pi) ** 2 * mass / charge
    return np.sign(curv) * np.sqrt(np.abs(curv) / C)


def freq_to_curv(freq, mass=None, charge=None, ion: Ion = None):
    """
    Curvature corresponding to the given secular frequency and ion

    Parameters
        freq: secular frequency [Hz]
        mass: mass of the ion [kg]
        charge: charge of the ion [C]
        ion (optional): ion species class specifying mass and charge

    Returns
        curv: potential curvature [V/m^2]
    """
    if ion is not None:
        mass = ion.mass
        charge = ion.charge
    C = (2 * np.pi) ** 2 * mass / charge
    return C * np.sign(freq) * freq**2


def field_to_shift(E, mass=None, charge=None, ion: Ion = None):
    """
    Shift in position produced by the electric field E
    in a potential well of 1 MHz

    Parameters
        E: electric field [V/m]
        mass: mass of the ion [kg]
        charge: charge of the ion [C]
        ion (optional): ion species class specifying mass and charge

    Returns
        shift: position shift [m]
    """
    curv = freq_to_curv(1e6, mass, charge, ion)
    return E / curv


def shift_to_field(dx, mass=None, charge=None, ion: Ion = None):
    """
    Electric field producing a shift in position dx
    in a potential well of 1 MHz

    Parameters
        E: electric field [V/m]
        mass: mass of the ion [kg]
        charge: charge of the ion [C]
        ion (optional): ion species class specifying mass and charge

    Returns
        shift: position shift [m]
    """
    curv = freq_to_curv(1e6, mass, charge, ion)
    return dx * curv


def get_hessian(axial, split, tilt, freq_pseudo, mass, charge):
    # v_ax, v_split, v_tilt = get_voltage_params(axial, split, tilt, freq_pseudo)[:3] * C * 1e12
    v_ax, v_split, v_tilt = freq_to_curv(np.asarray([axial, split, tilt]), mass, charge)
    v_ps = freq_to_curv(freq_pseudo)
    a = v_ps - v_ax / 2

    # TODO This is brutal. There should be a way to vectorize it but I'm lazy
    target_hessian = np.stack(
        [
            [[_v_ax, 0, 0], [0, _a + _v_split, _v_tilt], [0, _v_tilt, _a - _v_split]]
            for _v_ax, _a, _v_split, _v_tilt in zip(v_ax, a, v_split, v_tilt)
        ]
    )
    return target_hessian


def get_hessian_dc(axial, split, mass, charge):
    # force theta = 45, split is approximate
    v_ax = freq_to_curv(axial, mass, charge)
    b = (2 * np.pi) ** 2 * mass / charge * split * 5.5e6
    target_hessian = np.stack(
        [[[_v_ax, 0, 0], [0, -_v_ax / 2, _b], [0, _b, -_v_ax / 2]] for _v_ax, _b, in zip(v_ax, b)]
    )
    return target_hessian


# def get_voltage_params(axial, split, tilt, freq_pseudo):
#     if not freq_pseudo:
#         return np.ones((len(axial), 3, 3)) * np.nan
#     tilt = tilt * np.pi / 180
#     # assert -np.pi / 2 < tilt <= np.pi / 2, "Tilt angle must be within (-90, 90] degrees"
#     v_ax = freq_to_curv(axial)
#     v_ps = freq_to_curv(freq_pseudo)
#     a = v_ps - v_ax / 2
#     nu0 = curv_to_freq(a - freq_to_curv(split / 2))
#     lamb = C * nu0 * split
#     # nu1, nu2 = curv_to_freq(a + lamb) * 1e-6, curv_to_freq(a - lamb) * 1e-6
#     # print(f"Transverse mode freqs: {nu1:.3f}, {nu2:.3f} MHz (split: {nu1 - nu2:.3f})")

#     v_split = 2 * lamb * np.cos(tilt)**2 - lamb
#     v_tilt = np.sign(tilt) * np.sqrt(lamb**2 - v_split**2)

#     return np.asarray([v_ax, v_split, v_tilt, 0, 0, 0]) / C / 1e12
