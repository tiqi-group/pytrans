#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 03/2023
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
from nptyping import NDArray, Shape, Float
from typing import List
from tabulate import tabulate

from pytrans.ions import Ion, atomic_mass, elementary_charge
from pytrans.conversion import field_to_shift
from colorama import init as colorama_init, Fore
from scipy.optimize import OptimizeResult

colorama_init(autoreset=True)


def _color_str(color, str):
    return f"{color}{str:s}{Fore.RESET}"


def _diagonalize_hessian(ions: List[Ion], hessian: NDArray[Shape["L, L"], Float]):  # noqa
    N, d = len(ions), 3
    masses_amu = np.asarray([ion.mass_amu for ion in ions])
    masses = np.repeat(masses_amu, d)
    # H_w = 1 / np.sqrt(np.outer(masses, masses)) * hess  # this results in mass-weighted normal modes
    H_w = 1 / masses.reshape(-1, 1) * hessian  # standard normal modes
    h, v = np.linalg.eig(H_w)

    sort = np.abs(h).argsort()
    h = h[sort]  # shape: (3N,)
    freqs = np.sign(h) * np.sqrt(elementary_charge / atomic_mass * np.abs(h)) / 2 / np.pi
    modes = v.T[sort].reshape(N * d, N, d)  # shape: (3N, N, d)
    return freqs, modes


class Results:
    def __init__(self):
        self._printoptions = dict()
        self.set_printoptions()

    def set_printoptions(self, precision=4, format_char='g'):
        _locals = locals().copy()
        _locals.pop('self')
        self._printoptions.update(_locals)

    @staticmethod
    def _to_json_filter(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Ion):
            return str(value)
        elif isinstance(value, list) and isinstance(value[0], Ion):
            return [str(ion) for ion in value]
        elif isinstance(value, Results):
            return value.to_json()
        elif isinstance(value, OptimizeResult):
            return {key: Results._to_json_filter(value) for key, value in value.items()}
        else:
            return value

    def to_json(self):
        _json = {key: self._to_json_filter(value) for key, value in self.__dict__.items()}
        _json['repr'] = str(self)
        return _json


class ModeSolverResults(Results):
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

    def __init__(self, ions: List[Ion], x0, x_eq, fun, jac, hess, trap_pot, minimize_results, title=''):
        super().__init__()
        mode_freqs, mode_vectors = _diagonalize_hessian(ions, hess)
        self.x0 = x0
        self.ions = ions
        self.x_eq = x_eq
        self.fun = fun
        self.jac = jac
        self.hess = hess
        self.trap_pot = trap_pot
        self.mode_freqs = mode_freqs
        self.mode_vectors = mode_vectors
        self.minimize_results = minimize_results
        self.title = title

    def __repr__(self):
        headers = ['Freq [MHz]']
        for ion in self.ions:
            headers += ['', f"{ion}", '']
        L = 3 * len(self.ions)
        data = np.concatenate([self.mode_freqs.reshape(-1, 1) * 1e-6, self.mode_vectors.reshape(L, L)], axis=1)
        floatfmt = f".{self._printoptions['precision']}{self._printoptions['format_char']}"
        modes_table = tabulate(data, headers=headers, floatfmt=floatfmt)

        lines = []
        with np.printoptions(suppress=True, precision=self._printoptions['precision']):
            lines.append(_color_str(Fore.YELLOW, f"--------------\nMode solver analysis for ion crystal {self.ions}: {self.title}"))
            lines.append(_color_str(Fore.YELLOW, 'Equilibrium positions [um]'))
            lines.append(np.array_str(self.x_eq * 1e6))
            lines.append(_color_str(Fore.YELLOW, 'Normal modes'))
            lines.append(modes_table)
            lines.append("")
        return '\n'.join(lines)


class AnalysisResults(Results):

    def __init__(self, ion: Ion, x_eq, fun, jac, hess, minimize_results, title='', mode_solver_results=None):
        super().__init__()
        mode_freqs, mode_vectors = _diagonalize_hessian([ion], hess)
        mode_vectors = np.squeeze(mode_vectors)
        self.ion = ion
        self.x_eq = x_eq
        self.fun = fun
        self.jac = jac
        self.hess = hess
        self.shift = field_to_shift(jac, ion=ion)
        self.mode_freqs = mode_freqs
        self.mode_vectors = mode_vectors
        # TODO: x-axis specific
        self.mode_angle = np.arctan2(mode_vectors[2, 2], mode_vectors[2, 1]) * 180 / np.pi
        self.minimize_results = minimize_results
        self.mode_solver_results = mode_solver_results
        self.title = title

    def __repr__(self):
        lines = []
        with np.printoptions(suppress=True, precision=self._printoptions['precision']):
            lines.append(_color_str(Fore.YELLOW, f"--------------\nTrap potential analysis for ion {self.ion}: {self.title}"))
            lines.append(_color_str(Fore.YELLOW, 'Equilibrium position [um]'))
            lines.append(np.array_str(self.x_eq * 1e6))
            lines.append(_color_str(Fore.YELLOW, 'Gradient [V/m]'))
            lines.append(np.array_str(self.jac))
            lines.append(_color_str(Fore.YELLOW, 'Displacement at 1 MHz [um]'))
            lines.append(np.array_str(self.shift * 1e6))
            lines.append(_color_str(Fore.YELLOW, 'Hessian [V/m2]'))
            lines.append(np.array_str(self.hess))
            lines.append(_color_str(Fore.YELLOW, "Normal mode frequencies [MHz]"))
            formatter_freqs = {'float': lambda x: f"{x * 1e-6:g}" if x > 0 else _color_str(Fore.RED, f"{x * 1e-6:g}")}
            with np.printoptions(formatter=formatter_freqs):
                lines.append(np.array_str(self.mode_freqs))
            lines.append(_color_str(Fore.YELLOW, 'Eigenvectors'))
            formatter_modes = {'float': lambda x: _color_str(Fore.GREEN, f"{x:.3g}") if abs(x) > 0.9 else f"{x:.3g}"}
            with np.printoptions(formatter=formatter_modes):
                lines.append(np.array_str(self.mode_vectors))
            lines.append(_color_str(Fore.YELLOW, f"Tilt angle of mode 2 ({self.mode_freqs[2] * 1e-6:.2f}): {self.mode_angle:.2f}°"))
            if self.mode_solver_results is not None:
                lines.append(str(self.mode_solver_results))
            else:
                lines.append("")
        return '\n'.join(lines)
