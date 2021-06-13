#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 04 2021
# Author: carmelo <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np
import cvxpy as cx

from scipy.constants import atomic_mass, elementary_charge, pi
from analytic import potentialsDC
from analytic import gradientsDC
from analytic import hessiansDC

from analytic import pseudoPotential

mass = 40 * atomic_mass

# DC curvature to have a 1 MHz trap for 40Ca+
a0 = mass * (2 * pi * 1e6)**2 / elementary_charge
b0 = a0 * 1e-6


def electrode_potential(x, y, z, V, index):
    return V * getattr(potentialsDC, f"E{index}")(x, y, z)


def tot_potential(x, y, z, V, electrode_indices):
    tot = [
        electrode_potential(x, y, z, v, ix)
        for v, ix in zip(V, electrode_indices)
    ]
    return sum(tot)


def tot_potential_ps(x, y, z, V, electrode_indices, Vrf, Omega_rf):
    return tot_potential(x, y, z, V, electrode_indices) + pseudoPotential.ps0(
        x, y, z, Vrf, Omega_rf)


def tot_gradient(x, y, z, V, electrode_indices):
    tot = [
        v * getattr(gradientsDC, f"E{ix}")(x, y, z)
        for v, ix in zip(V, electrode_indices)
    ]
    return sum(tot)


def tot_gradient_ps(x, y, z, V, electrode_indices, Vrf, Omega_rf):
    return tot_gradient(x, y, z, V, electrode_indices) + pseudoPotential.ps1(
        x, y, z, Vrf, Omega_rf)


def tot_hessian(x, y, z, V, electrode_indices):
    tot = [
        v * getattr(hessiansDC, f"E{ix}")(x, y, z)
        for v, ix in zip(V, electrode_indices)
    ]
    return sum(tot)


def tot_hessian_ps(x, y, z, V, electrode_indices, Vrf, Omega_rf):
    return tot_hessian(x, y, z, V, electrode_indices) + pseudoPotential.ps2(
        x, y, z, Vrf, Omega_rf)


def hessian_pot(x, y, z, r0, M, E):
    x0, y0, z0 = r0
    xyz = np.stack(np.broadcast_arrays(x - x0, y - y0, z - z0))
    return 0.5 * a0 * np.einsum('a...,ab,b...', xyz, M, xyz) - b0 * np.einsum(
        'a,a...', E, xyz)


def make_roi(r0, electrode_indices, roi=(400, 100, 100), shape=(50, 30, 30)):
    x0, y0, z0 = r0

    def __roi(L, n):
        ls = np.linspace(-L, L, n) * 1e-6
        s = L / 3 * 1e-6
        return ls, s

    # let's just be lazy here
    lx, ly, lz = roi
    nx, ny, nz = shape
    x_roi, sx = __roi(lx, nx)
    y_roi, sy = __roi(ly, ny)
    z_roi, sz = __roi(lz, nz)

    X, Y, Z = np.meshgrid(x_roi, y_roi, z_roi, indexing='ij')

    roi_weight = np.exp(-(X**2 / sx**2 + Y**2 / sy**2 + Z**2 / sz**2) / 2).ravel()

    def moments_matrix(x, y, z):
        moments = [
            getattr(potentialsDC, f"E{ix}")(x, y, z).ravel()
            for ix in electrode_indices
        ]
        return np.stack(moments, axis=-1)

    moments = moments_matrix(X + x0, Y + y0, Z + z0)

    return (X, Y, Z), moments, roi_weight


def roi_cost(u, M, E, XYZ, moments, roi_weight, uoffs=0):
    target = hessian_pot(*XYZ, r0=[0, 0, 0], M=M, E=E).ravel()
    cost = cx.sum_squares(cx.multiply(roi_weight,
                                      moments @ u - uoffs - target))
    return cost


def hessian_cost(u, M, E, r0, electrode_indices):
    cost = cx.sum_squares(tot_hessian(*r0, u, electrode_indices) / a0 - M)
    cost += cx.sum_squares(tot_gradient(*r0, u, electrode_indices) / b0 + E)
    return cost


def solve_voltages(r0, M, E, electrode_indices,
                   y_sign=1,
                   hessian_weight=1,
                   XYZ=None,
                   moments=None,
                   roi_weight=None,
                   u_start=None,
                   dv=1e-2):
    ll = len(electrode_indices)

    if hessian_weight == 1:
        u = cx.Variable((ll, ), name='voltages')
    else:
        u = cx.Variable((ll + 1, ), name='voltages')
        u, uoffs = u[:-1], u[-1]

    if u_start is not None:
        u.value = u_start
    assert hessian_weight >= 0 and hessian_weight <= 1

    if hessian_weight == 0:
        cost = roi_cost(u, M, E, XYZ, moments, roi_weight, uoffs)
    elif hessian_weight == 1:
        cost = hessian_cost(u, M, E, r0, electrode_indices)
    else:
        ch = hessian_cost(u, M, E, r0, electrode_indices)
        cr = roi_cost(u, M, E, XYZ, moments, roi_weight, uoffs)
        cost = hessian_weight * ch + (1 - hessian_weight) * cr

    # if ww is not None:
    #     cost += cx.sum_squares(cx.multiply(ww, u))

    constraints = [-10 <= u[:-1], u[:-1] <= 10]
    constraints += [u[:ll // 2] == y_sign * u[ll // 2:]]
    if u_start is not None:
        constraints += [cx.sum_squares(u_start - u) <= dv**2 * ll]

    objective = cx.Minimize(cost)
    prob = cx.Problem(
        objective,
        constraints,
    )

    prob.solve(warm_start=True, solver='MOSEK', verbose=False)
    return u.value
