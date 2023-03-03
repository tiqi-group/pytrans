#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 05/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from pytrans.analysis import coulomb as pcoul
from pytrans.conversion import curv_to_freq
from pytrans.abstract_model import AbstractTrapModel

from colorama import init as colorama_init, Fore

colorama_init(autoreset=True)


def solver1d_two_ions(trap, voltages, r0, x_range, pseudo=True, dx=5e-6, minimize_options=dict()):
    q = trap.ion.charge
    x1, y1, z1 = r0
    x0 = np.asarray([x1 - dx, x1 + dx])
    assert x_range[0] < x1 and x1 < x_range[1]

    def fun(x0):
        return trap.potential(voltages, x0, y1, z1, pseudo).sum() + q * pcoul.coulomb_pot(*x0)

    def jac(x0):
        return trap.gradient(voltages, x0, y1, z1, pseudo)[0] + q * pcoul.coulomb_grad(*x0)

    def hess(x0):
        return np.diag(trap.hessian(voltages, x0, y1, z1, pseudo)[0, 0]) + q * pcoul.coulomb_hess(*x0)

    bounds = [(x_range[0], x_range[1])] * 2
    res = minimize(fun, x0, method='TNC', jac=jac, bounds=bounds, tol=q * pcoul.kappa, options=minimize_options)
    res.hess = hess(res.x)
    return res


def analyse_potential_two_ions(trap: AbstractTrapModel, voltages: ArrayLike, r0: ArrayLike, roi=(-200, 200),
                               plot=True, ax=None, title='',
                               pseudo=True, minimize_options=dict(), verbose=True):

    if verbose:
        print('--------------\n' + Fore.YELLOW + f"Analyse two ion modes: {title}")

    x1, y1, z1 = r0
    x_range = (x1 + roi[0] * 1e-6, x1 + roi[1] * 1e-6)
    res = solver1d_two_ions(trap, voltages, r0, x_range,
                            pseudo=pseudo, minimize_options=minimize_options)
    h, ei = np.linalg.eig(res.hess)
    freqs = curv_to_freq(h, ion=trap.ion)
    dist = res.x[1] - res.x[0]
    alpha = voltages @ trap.dc_hessians(*r0)[:, 0, 0] / 2
    beta = voltages @ trap.dc_fourth_order(*r0) / 12

    if verbose:
        with np.printoptions(formatter={'float': lambda x: f"{x:.3f}"}):
            print(f"{Fore.YELLOW}Two ion distance [um]:{Fore.RESET} {dist * 1e6: .2f}")
            print(f"{Fore.YELLOW}Two ion normal modes: {Fore.RESET}{freqs * 1e-6} MHz")
            print(Fore.YELLOW + 'Eigenvectors')
            print(ei)
        # print(Fore.YELLOW + 'Minimize results')
        # print(res)

    results = dict(
        x1=res.x[0],
        x2=res.x[1],
        dist=dist,
        f_com=freqs.min(),
        f_str=freqs.max(),
        freqs=freqs,
        alpha=alpha,
        beta=beta,
        eigenvalues=h,
        eigenvectors=ei,
        res=res
    )

    if not plot:
        return results

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    x = np.linspace(roi[0], roi[1], 100) * 1e-6 + x1
    pot = trap.potential(voltages, x, y1, z1, pseudo=pseudo)
    p1 = trap.potential(voltages, x1, y1, z1, pseudo=pseudo)

    fig.suptitle(title)
    ax.plot(x * 1e6, pot)
    ax.set_ylim(ax.get_ylim())
    ax.plot(x * 1e6, p1 + alpha * (x - x1)**2 + beta * (x - x1)**4)

    p2 = trap.potential(voltages, res.x, y1, z1, pseudo=pseudo)
    ax.plot(res.x * 1e6, p2, 'dr')

    return results
