#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 10/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
from scipy.constants import pi, elementary_charge, epsilon_0
from scipy.optimize import minimize

from numpy.typing import NDArray
from typing import List
from pytrans.typing import Coords

from pytrans.abstract_model import AbstractTrapModel
from pytrans.ions import Ion, atomic_mass

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
    return -kappa * np.sum(r_ab / r[..., None] ** 3, axis=1)


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
    H[np.diag_indices(N)] = -H.sum(axis=1)
    return H


def mode_solver(
    trap: AbstractTrapModel,
    voltages: NDArray,
    ions: List[Ion],
    x0: Coords,
    bounds=None,
    sort_axis=None,
    minimize_options=dict(),
) -> ModeSolverResults:
    N, d = x0.shape
    # ions = [ions] * N if isinstance(ions, Ion) else ions
    masses_amu = np.asarray([ion.mass_amu for ion in ions])

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

    res = minimize(fun, x0.ravel(), method="TNC", jac=jac, bounds=bounds, options=options)
    x_eq = res.x.reshape((N, d))
    trap_pot = trap.potential(voltages, *x_eq.T, masses_amu)
    hess = hess(res.x)
    mode_freqs, mode_vectors = diagonalize_hessian(ions, hess)

    if sort_axis is not None:
        ix = np.argsort(x_eq[:, 0])
        x_eq = x_eq[ix]
        mode_vectors = mode_vectors[:, ix, :]

    result = ModeSolverResults(
        ions=ions,
        x0=x0,
        x_eq=x_eq,
        fun=res.fun,
        jac=res.jac,
        hess=hess,
        mode_freqs=mode_freqs,
        mode_vectors=mode_vectors,
        trap_pot=trap_pot,
        minimize_results=res,
    )

    return result


def _ravel_coords(*args):
    args = np.broadcast_arrays(*args)
    shape = args[0].shape
    args = list(map(np.ravel, args))
    X = np.stack(args, axis=1).astype(float)
    return shape, X


def diagonalize_hessian(ions: List[Ion], hessian: NDArray):  # noqa
    N, d = len(ions), 3
    masses_amu = np.asarray([ion.mass_amu for ion in ions])
    masses = np.repeat(masses_amu, d)
    H_w = 1 / np.sqrt(np.outer(masses, masses)) * hessian  # this results in mass-weighted normal modes
    # H_w = 1 / masses.reshape(-1, 1) * hessian  # standard normal modes
    h, v = np.linalg.eigh(H_w)

    sort = np.abs(h).argsort()
    h = h[sort]  # shape: (3N,)
    freqs = np.sign(h) * np.sqrt(elementary_charge / atomic_mass * np.abs(h)) / 2 / np.pi
    modes = v.T[sort].reshape(N * d, N, d)  # shape: (3N, N, d)
    return freqs, modes


def project_on_single_ion_modes(
    mode_vectors: NDArray,  # noqa: F821
    single_ion_modes: NDArray = np.eye(3),
    keys: List[str] = ["x", "y", "z"],
):
    """
    Project the normal modes of a string of trapped ions on single-ion eigenmodes.

    Args:
        mode_vectors (np.ndarray):
            A (3N, N, 3) array containing the mode participation eigenvectors
            of the normal modes of a string of N trapped ions.
        single_ion_modes (np.ndarray, optional):
            A (3, 3) orthogonal matrix with the mode orientations
            of a single particle. Defaults to np.eye(3).
        keys (List[str], optional): A list of three strings naming the modes. Defaults to ['x', 'y', 'z'].

    Returns:
        mode_vectors_projected : ndarray, shape (3N, N)
            projections of mode_vectors on the target single ion mode
        The second element is a dictionary mapping the mode names ('x', 'y', or 'z')
          to the indices of the modes in `mode_vectors` that correspond to each name.

    Example:
    >>> mode_vectors = np.random.rand(9, 3, 3)
    >>> single_ion_modes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> keys = ['x', 'y', 'z']
    >>> project_on_single_ion_modes(mode_vectors, single_ion_modes, keys)
    (array([...]), {'x': array([...]), 'y': array([...]), 'z': array([...])})
    """
    # projections of normal modes on single-ion eigenmodes
    proj = abs(np.einsum("Mai,mi", mode_vectors, single_ion_modes)).sum(1)
    mode1_index = np.argmax(proj, axis=1)

    mode_vectors_projected = np.asarray(
        [mode_vectors[j] @ single_ion_modes[mode1_index[j]] for j in range(len(mode_vectors))]
    )

    mode_labels = {}
    keys = "xyz" if keys is None else keys
    for j, key in enumerate(keys):
        mode_labels[key] = np.where(mode1_index == j)[0]

    return mode_vectors_projected, mode_labels


def init_crystal(r0: NDArray, dx: float, n_ions: int, axis=0, randomize=True) -> Coords:
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
    X = np.zeros((n_ions, 3), dtype=float)
    X[:] = r0
    X[:, axis] += np.linspace(-n_ions / 2 * dx, n_ions / 2 * dx, n_ions)
    if randomize:
        X += np.random.randn(n_ions, 3) * dx * 1e-2
    return X
