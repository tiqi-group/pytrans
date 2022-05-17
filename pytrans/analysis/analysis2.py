#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 05/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.constants import epsilon_0

from pytrans.conversion import curv_to_freq
from pytrans.abstract_model import AbstractTrap

from colorama import init as colorama_init, Fore

colorama_init(autoreset=True)


def solver1d_two_ions(trap, voltages, r0, x_range, pseudo=True, dx=5e-6, minimize_options=dict()):
    kappa = trap.ion.charge / 4 / np.pi / epsilon_0
    x1, y1, z1 = r0
    x0 = np.asarray([x1 - dx, x1 + dx])
    assert x_range[0] < x1 and x1 < x_range[1]

    def coulomb_pot(x1, x2):
        return kappa / (x2 - x1)

    def coulomb_grad(x1, x2):
        return kappa / (x2 - x1)**2 * np.asarray([1, -1])

    def coulomb_hess(x1, x2):
        return 2 * kappa / (x2 - x1)**3 * np.asarray([[1, -1], [-1, 1]])

    def fun(x0):
        return trap.potential(voltages, x0, y1, z1, pseudo).sum() + coulomb_pot(*x0)

    def jac(x0):
        return trap.gradient(voltages, x0, y1, z1, pseudo)[0] + coulomb_grad(*x0)

    def hess(x0):
        return np.diag(trap.hessian(voltages, x0, y1, z1, pseudo)[0, 0]) + coulomb_hess(*x0)

    bounds = [(x_range[0], x1), (x1, x_range[1])]
    res = minimize(fun, x0, method='TNC', jac=jac, bounds=bounds, tol=kappa, options=minimize_options)
    res.hess = hess(res.x)
    return res


def analyse_potential_two_ions(trap: AbstractTrap, voltages: ArrayLike, r0: ArrayLike, roi=(-200, 200),
                               plot=True, title='',
                               pseudo=False, minimize_options=dict(), verbose=True):

    if verbose:
        print('--------------\n' + Fore.YELLOW + f"Analyse two ion modes: {title}")

    x1, y1, z1 = r0
    x_range = (x1 + roi[0] * 1e-6, x1 + roi[1] * 1e-6)
    res = solver1d_two_ions(trap, voltages, r0, x_range,
                            pseudo=pseudo, minimize_options=minimize_options)
    h, ei = np.linalg.eig(res.hess)

    if verbose:
        with np.printoptions(formatter={'float': lambda x: f"{x:.3f}"}):
            print(f"{Fore.YELLOW}Two ion distance [um]:{Fore.RESET} {abs(np.diff(res.x)[0]) * 1e6: .2f}")
            print(f"{Fore.YELLOW}Two ion normal modes: {Fore.RESET}{curv_to_freq(h, ion=trap.ion) * 1e-6} MHz")
            print(Fore.YELLOW + 'Eigenvectors')
            print(ei)
        print(Fore.YELLOW + 'Minimize results')
        print(res)

    x = np.linspace(roi[0], roi[1], 100) * 1e-6 + x1
    pot = trap.potential(voltages, x, y1, z1, pseudo=pseudo)
    p1 = trap.potential(voltages, x1, y1, z1, pseudo=pseudo)
    alpha = voltages @ trap.dc_hessians(*r0)[:, 0, 0] / 2
    beta = voltages @ trap.dc_fourth_order(*r0) / 12

    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.plot(x * 1e6, pot)
    ax.set_ylim(ax.get_ylim())
    ax.plot(x * 1e6, p1 + alpha * (x - x1)**2 + beta * (x - x1)**4)

    p2 = trap.potential(voltages, res.x, y1, z1, pseudo=pseudo)
    ax.plot(res.x * 1e6, p2, 'dr')

    return res
