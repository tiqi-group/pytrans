#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 10/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
from scipy.constants import pi, elementary_charge, epsilon_0
from scipy.optimize import minimize

from typing import List
from nptyping import NDArray, Shape, Float
from pytrans.typing import Coords

from pytrans.abstract_model import AbstractTrapModel
from pytrans.ions import Ion, atomic_mass
from pytrans.conversion import curv_to_freq

from pytrans.analysis.results import ModeSolverResults

kappa = elementary_charge / 4 / pi / epsilon_0


def distances(X: Coords):
    """distances between particles

    Args:
        X (array, shape (N, d)): particle positions

    Returns:
        r_ab (array, shape (N, N, d)): particle distances, where
          r_ab[a, b, j] = (X[a] - X[b])[j]
    """
    N, d = X.shape
    r = np.empty((N, N, d))
    for j, w in enumerate(X.T):
        r[:, :, j] = w[:, None] - w
    return r


def coulomb_potential(X: Coords):
    """coulomb_potential

    Args:
        X (array, shape (N, d)): particle positions

    Returns:
        U (float): total Coulomb potential
          U = sum_ab kappa / abs(X[a] - X[b])s
    """
    if X.shape[0] == 1:
        return 0
    r = distances(X)
    r = np.sqrt(np.sum(r**2, axis=-1))
    pot = kappa * np.sum(1 / r[np.triu_indices(len(r), k=1)])
    return pot


def coulomb_gradient(X: Coords):
    """coulomb_gradient

    Args:
        X (array, shape (N, d)): particle positions

    Returns:
        f (array, shape (N, d)): gradient of U = coulomb_potential(X)
          f[a, j] = d U / d X[a, j]
          or, -f[a] is the total coulomb force on particle a
    """
    r_ab = distances(X)  # shape (N, N, d)
    r = np.sqrt(np.sum(r_ab**2, axis=-1))  # shape (N, N), diag == 0
    np.fill_diagonal(r, np.inf)
    return - kappa * np.sum(r_ab / r[..., None]**3, axis=1)


def coulomb_hessian(X: Coords):
    """coulomb_hessian

    Args:
        X (array, shape (N, d)): particle positions

    Returns:
        H (array, shape (N, N, d, d)): gradient of U = coulomb_potential(X)
          H[a, b, i, j] = d^2 U / d X[a, i] d X[b, j]
          or, -H[a, b] is the gradient in the coulomb force on particle a
          due to displacing particle b
    """
    N, d = X.shape
    r_ab = distances(X)  # shape (N, N, d)
    r = np.sqrt(np.sum(r_ab**2, axis=-1))  # shape (N, N), diag == 0
    np.fill_diagonal(r, np.inf)
    r = r[:, :, None, None]
    d_ij = np.eye(d)[None, None, :, :]
    H = kappa * (d_ij / r**3 - 3 * r_ab[:, :, :, None] * r_ab[:, :, None, :] / r**5)
    H[np.diag_indices(N)] = - H.sum(axis=1)
    return H


class HarmonicTrap:

    def __init__(self, fx, fy, fz, ion: Ion, field=[0, 0, 0],
                 tilt_xy=0, tilt_xz=0, tilt_yz=0):
        wx2, wy2, wz2 = (2 * pi * fx)**2, (2 * pi * fy)**2, (2 * pi * fz)**2
        c_x = ion.mass / ion.charge * wx2
        c_dc = ion.mass / ion.charge * (wy2 - wz2) / 2
        m_c_rf = ion.mass**2 / ion.charge * (wx2 + wy2 + wz2) / 2

        self.rf_null_coords = (None, 0, 0)

        self._H_dc = np.asarray([
            [c_x, tilt_xy, tilt_xz],
            [tilt_xy, c_dc - c_x / 2, tilt_yz],
            [tilt_xz, tilt_yz, -c_dc - c_x / 2]
        ])

        self._m_H_rf = np.asarray([
            [0, 0, 0],
            [0, m_c_rf, 0],
            [0, 0, m_c_rf]
        ])

        self._E = np.asarray(field)

    def _H(self, mass_amu):
        mass = atomic_mass * np.atleast_1d(mass_amu).reshape(-1, 1, 1)
        return self._H_dc.reshape(1, 3, 3) + self._m_H_rf / mass

    # def _X(self, x, y, z):
    #     return np.stack([x, y, z], axis=-1)
    def _ravel_coords(self, *args):
        args = np.broadcast_arrays(*args)
        shape = args[0].shape
        args = list(map(np.ravel, args))
        X = np.stack(args, axis=1).astype(float)
        return shape, X

    def potential(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        H = self._H(mass_amu)
        pot = 0.5 * np.einsum('...i,...ij,...j', X, H, X) + \
            np.einsum('j,...j', self._E, X)
        return pot.reshape(shape)

    def gradient(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        H = self._H(mass_amu)
        grad = np.einsum('...ij,...j', H, X) + self._E.reshape(1, -1)
        return grad.reshape(shape + (3,))

    def hessian(self, voltages, x, y, z, mass_amu, pseudo=True):
        shape, X = self._ravel_coords(x, y, z)
        H = self._H(mass_amu)
        return H.reshape(shape + (3, 3))

    def trap_frequencies(self, ion: Ion):
        curv = np.diag(self._H(ion.mass_amu))
        return curv_to_freq(curv, ion=ion)


def mode_solver(trap: AbstractTrapModel, voltages: NDArray, ions: List[Ion],
                x0: Coords, bounds=None, minimize_options=dict()) -> ModeSolverResults:
    N, d = x0.shape
    # ions = [ions] * N if isinstance(ions, Ion) else ions
    masses_amu = np.asarray([ion.mass_amu for ion in ions])

    def fun(X):
        _X = X.reshape(N, d)
        return trap.potential(voltages, *_X.T, masses_amu).sum() + coulomb_potential(_X)

    def jac(X):
        _X = X.reshape(N, d)
        grad = trap.gradient(voltages, *_X.T, masses_amu) + \
            coulomb_gradient(_X)
        grad = grad.ravel()
        return grad

    def hess(X):
        """Total mass-dependent hessian

        Returns:
            H (array, shape: (Nd, Nd)):
                In the typical case d = 3, it is a (3N, 3N) matrix where coordinates
                are sorted like (x1, y1, z1, x2, y2, z2, ..., xN, yN, zN)
                for ions 1 ... N
        """
        _X = X.reshape(N, d)
        hess = coulomb_hessian(_X)  # shape (N, N, d, d)
        trap_hess = trap.hessian(
            voltages, *_X.T, masses_amu)  # shape (N, d, d)
        hess[np.diag_indices(N, ndim=2)] += trap_hess  # add it in blocks
        hess = np.swapaxes(hess, 1, 2).reshape((N * d, N * d))
        return hess

    # eta = 1e-9
    bounds = list(bounds) * N if bounds is not None else None

    options = dict(
        # maxCGit=0,
        accuracy=1e-8,
        xtol=1e-8,
        ftol=kappa,
        # gtol=kappa / 1e-5,
        # scale=1e-6 * np.ones((N * d,)),
        # maxfun=300 * N * d
    )
    options.update(minimize_options)

    res = minimize(fun, x0.ravel(), method='TNC', jac=jac,
                   bounds=bounds, options=options)
    x_eq = res.x.reshape((N, d))
    trap_pot = trap.potential(voltages, *x_eq.T, masses_amu)
    hess = hess(res.x)
    mode_freqs, mode_vectors = diagonalize_hessian(ions, hess)

    result = ModeSolverResults(ions=ions, x0=x0, x_eq=x_eq,
                               fun=res.fun, jac=res.jac, hess=hess,
                               mode_freqs=mode_freqs, mode_vectors=mode_vectors,
                               trap_pot=trap_pot,
                               minimize_results=res)

    return result


def _ravel_coords(*args):
    args = np.broadcast_arrays(*args)
    shape = args[0].shape
    args = list(map(np.ravel, args))
    X = np.stack(args, axis=1).astype(float)
    return shape, X


def diagonalize_hessian(ions: List[Ion], hessian: NDArray[Shape["L, L"], Float]):  # noqa
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


def init_crystal(r0: NDArray[Shape["3"], Float], dx: float, n_ions: int, axis=0, randomize=True) -> Coords:
    """initialize positions of particles in a 1D crystal
    equally spaced by dx along the specified axis

    Args:
        r0 (array_like, shape (3,)): (x, y, z) position of the center of mass of the crystal
        dx (float): particle spacing
        n_ions (int): number of particles
        axis (int, default = 0): crystal axis

    Returns:
        Coords (array, shape: (n_ions, 3)): particle positions in a crystal
    """
    x0, y0, z0 = r0
    X = np.zeros((n_ions, 3), dtype=float)
    X[:, axis] = np.linspace(-n_ions / 2 * dx, n_ions / 2 * dx, n_ions) + x0
    X[:, axis - 1] = y0
    X[:, axis - 2] = z0
    if randomize:
        X += np.random.randn(n_ions, 3) * dx * 1e-2
    return X
