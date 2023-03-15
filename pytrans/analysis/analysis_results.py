#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 03/2023
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
from typing import List
from tabulate import tabulate

from pytrans.ions import Ion, atomic_mass, elementary_charge

from colorama import init as colorama_init, Fore

colorama_init(autoreset=True)


def _color_str(color, str):
    return f"{color}{str:s}{Fore.RESET}"


class AnalysisResults:
    """ Analysis results

    Attributes:
        x0 (array, shape (N, 3)) initial positions
        ions (list of Ions, len N): Ion objects
        x_eq (array, shape (N, 3)): equilibrium positions
        fun (float): total potential at minimum
        jac (array, slape (N,)): total gradient at minimum
        hess (array, shape (3N, 3N)): mass-dependent hessian at minimum
        mode_freqs (array, shape (3N,)): normal modes frequencies in Hertz
        mode_vectors (array, shape (3N, N, 3)): normal modes eigenvectors
            mode_vectors[n, k, :] are the (x, y, z) components of
            mode n on ion k
        minimize_result: minimization result returned by scipy.minimize
    """

    def __init__(self, ions: List[Ion], x0, x_eq, fun, jac, hess, minimize_result):
        N, d = x0.shape
        masses_amu = np.asarray([ion.mass_amu for ion in ions])
        masses = np.repeat(masses_amu, d)
        H_w = 1 / np.sqrt(np.outer(masses, masses)) * hess  # this results in mass-weighted normal modes
        h, v = np.linalg.eig(H_w)

        sort = np.abs(h).argsort()
        h = h[sort]  # shape: (3N,)
        freqs = np.sign(h) * np.sqrt(elementary_charge / atomic_mass * np.abs(h)) / 2 / np.pi
        em_matrix = v[:, sort]
        v = em_matrix.T.reshape(N * d, N, d)  # shape: (3N, N, d)

        self.x0 = x0
        self.ions = ions
        self.x_eq = x_eq
        self.fun = fun
        self.jac = jac
        self.hess = hess
        self.mode_freqs = freqs
        self.mode_vectors = v
        self.em_matrix = em_matrix
        self.minimize_result = minimize_result
        self._printoptions = dict()
        self.set_printoptions()

    def set_printoptions(self, precision=3, format_char='g'):
        _locals = locals().copy()
        _locals.pop('self')
        self._printoptions.update(_locals)

    def __repr__(self):
        headers = ['Freq [MHz]']
        for ion in self.ions:
            headers += ['', f"{ion}", '']
        data = np.concatenate([self.mode_freqs.reshape(-1, 1) * 1e-6, self.em_matrix.T], axis=1)
        floatfmt = f".{self._printoptions['precision']}{self._printoptions['format_char']}"
        modes_table = tabulate(data, headers=headers, floatfmt=floatfmt)

        lines = []
        with np.printoptions(suppress=True):
            lines.append(_color_str(Fore.YELLOW, 'Equilibrium position [um]'))
            lines.append(np.array_str(self.x_eq * 1e6))
            if len(self.ions) <= 1:
                lines.append(_color_str(Fore.YELLOW, 'Gradient [V/m]'))
                lines.append(np.array_str(self.jac))
                lines.append(_color_str(Fore.YELLOW, 'Hessian [V/m2]'))
                lines.append(np.array_str(self.hess))
            # lines.append(repr(_color_str(Fore.YELLOW, "Normal mode frequencies [MHz]")))
            # formatter_freqs = {'float': lambda x: f"{x * 1e-6:g}" if x > 0 else _color_str(Fore.RED, f"{x * 1e-6:g}")}
            # with np.printoptions(formatter=formatter_freqs):
            #     lines.append(repr(freqs))
            # lines.append(repr(_color_str(Fore.YELLOW, 'Eigenvectors')))
            # formatter_modes = {'float': lambda x: _color_str(Fore.GREEN, f"{x:.3g}") if abs(x) > 0.9 else f"{x:.3g}"}
            # with np.printoptions(formatter=formatter_modes):
            #     lines.append(repr(vs))
            lines.append(_color_str(Fore.YELLOW, 'Normal modes'))
            lines.append(modes_table)
            # lines.append(repr(_color_str(Fore.YELLOW, f"Tilt angle of mode 2 ({freqs[2] * 1e-6:.2f}):") + f" ){angle:.2f}°"))
            lines.append("")
        return '\n'.join(lines)
