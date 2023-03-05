#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 10/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
from nptyping import NDArray, Shape, Float
from typing import Union, List
from scipy.constants import pi, elementary_charge, epsilon_0
from scipy.optimize import minimize

from pytrans.abstract_model import AbstractTrapModel
from pytrans.ions import Ion, atomic_mass

kappa = elementary_charge / 4 / pi / epsilon_0
Coords = NDArray[Shape["*, 3"], Float]


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


def mode_solver(trap: AbstractTrapModel, voltages: NDArray, ions: Union[Ion, List[Ion]],
                x0: Coords, bounding_box=None, minimize_options=dict()):
    N, d = x0.shape
    if isinstance(ions, list):
        masses_amu = np.asarray([ion.mass_amu for ion in ions])
    else:
        masses_amu = np.asarray([ions.mass_amu for _ in range(N)])

    def fun(X):
        _X = X.reshape(N, d)
        return trap.potential(voltages, *_X.T, masses_amu).sum() + coulomb_potential(_X)

    def jac(X):
        _X = X.reshape(N, d)
        grad = trap.gradient(voltages, *_X.T, masses_amu) + coulomb_gradient(_X)
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
        trap_hess = trap.hessian(voltages, *_X.T, masses_amu)  # shape (N, d, d)
        hess[np.diag_indices(N, ndim=2)] += trap_hess  # add it in blocks_type_
        hess = np.swapaxes(hess, 1, 2).reshape((N * d, N * d))
        return hess

    # eta = 1e-9
    bounds = list(bounding_box) * N if bounding_box else None

    options = dict(
        # maxCGit=0,
        xtol=1e-8,
        ftol=kappa,
        # gtol=kappa / 1e-5,
        # scale=1e-6 * np.ones((N * d,)),
        maxfun=300 * N * d
    )
    options.update(minimize_options)

    res = minimize(fun, x0.ravel(), method='TNC', jac=jac, bounds=bounds, options=options)

    x1 = res.x.reshape(N, d)  # equilibrium position
    H = hess(res.x)  # mass-dependent hessian, (3N, 3N)
    masses = np.repeat(masses_amu, 3) * atomic_mass
    H_w = 1 / np.sqrt(np.outer(masses, masses)) * H  # mass-weighted hessian
    h, v = np.linalg.eig(H_w)

    sort = np.abs(h).argsort()
    h = h[sort]  # shape: (3N,)
    freqs = np.sign(h) * (np.abs(h) * elementary_charge)**(1 / 2) / 2 / pi
    v = v[:, sort].T.reshape(N * d, N, d)  # shape: (3N, N, d)

    result = {
        'x_eq': x1,
        'mode_freqs': freqs,
        'mode_vectors': v,
        'res': res
    }
    return result


def _ravel_coords(*args):
    args = np.broadcast_arrays(*args)
    shape = args[0].shape
    args = list(map(np.ravel, args))
    X = np.stack(args, axis=1).astype(float)
    return shape, X


def init_crystal(r0: NDArray[Shape["3"], Float], dx: float, n_ions: int) -> Coords:
    """initialize positions of particles in a 1D crystal
    equally spaced by dx along the x axis

    Args:
        r0 (array_like, shape (3,)): (x, y, z) position of the center of mass of the crystal
        dx (float): particle spacing
        n_ions (int): number of particles

    Returns:
        Coords (array, shape: (n_ions, 3)): particle positions in a crystal
    """
    x0, y0, z0 = r0
    X = np.zeros((n_ions, 3), dtype=float)
    X[:, 0] = np.linspace(-n_ions / 2 * dx, n_ions / 2 * dx, n_ions) + x0
    X[:, 1] = y0
    X[:, 2] = z0
    return X
