#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 05/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np
import time
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from tqdm import tqdm
from .mode_solver import coulomb_gradient

from pytrans.abstract_model import AbstractTrapModel
from pytrans.ions import Ion
from pytrans.typing import Coords, Waveform
from typing import List

# length and time scales
from scipy.constants import atomic_mass as _m0
from scipy.constants import elementary_charge as _q0
_x0 = 1e-6
_t0 = 1e-6

_p0 = _m0 * _x0 / _t0
_E0 = _m0 * _x0 / _q0 / _t0**2


def simulate_waveform(trap: AbstractTrapModel, waveform: Waveform, ions: List[Ion], t, dt, x0: Coords, v0=None, bounds=None, slowdown=1, pseudo=True, solve_kw=dict()):

    N, d = x0.shape
    mass_amu = np.asarray([ion.mass_amu for ion in ions])
    unit_charge = np.asarray([ion.unit_charge for ion in ions])

    n_samples = len(waveform)
    if n_samples == 1:
        def waveform_t(t):
            return waveform[0]
    else:
        s = np.linspace(0, 1, n_samples)
        waveform_s = interp1d(s, waveform, axis=0,
                              bounds_error=False, fill_value=(waveform[0], waveform[-1]))

        def waveform_t(t):
            return waveform_s(t * _t0 / (dt * n_samples * slowdown))

    v0 = np.zeros_like(x0) if v0 is None else v0
    y0 = np.r_[x0.ravel() / _x0, v0.ravel() / _p0]

    def force(t, x0):
        _X = x0.reshape(N, d) * _x0
        vv = waveform_t((t))
        trap_gradient = trap.gradient(vv, *_X.T, mass_amu, pseudo=pseudo)
        cg = coulomb_gradient(_X)
        force = - unit_charge.reshape(-1, 1) * (trap_gradient + cg) / _E0
        return force.ravel()

    if bounds is not None:
        raise ValueError("Bounds termination not implemented")
    #     assert isinstance(bounds, (list, tuple)) and len(bounds) == 2
    #     events = []
    #     for j in range(N):
    #         def exit_event(t, y, *args):
    #             return (y[j] - bounds[0] / _x0) * (y[j] - bounds[1] / _x0)
    #         exit_event.terminal = True
    #         events.append(exit_event)
    # else:
    #     events = None

    def fun(t, y):
        # y = x1, x2, p1, p2
        return np.r_[y[N * d:] / np.repeat(mass_amu, d), force(t, y[:N * d])]

    def fun_pbar(t, y, pbar, state):
        # https://stackoverflow.com/a/62140877
        last_t, dt = state
        n = int((t - last_t) / dt)
        pbar.update(n)
        state[0] = last_t + dt * n
        return fun(t, y)

    t0, t1 = t[[0, -1]] / _t0
    state = [t0, (t1 - t0) / 1000]

    kw = dict(t_eval=t / _t0, dense_output=True, events=None, method='LSODA')
    kw.update(solve_kw)

    print("Exec simulate_waveform")
    ts = time.time()

    with tqdm(total=1000, unit="%") as pbar:
        sol = solve_ivp(fun_pbar, (t0, t1), y0, args=(pbar, state), **kw)

    te = time.time()
    elapsed = te - ts
    print(f"- simulate_waveform elapsed time: {elapsed * 1e3:.3f} ms")

    sol.t = sol.t * _t0
    sol.x = (sol.y[:N * d] * _x0).T.reshape(len(t), N, d)
    sol.p = (sol.y[N * d:] * _p0).T.reshape(len(t), N, d)

    # def kin(p):
    #     k = p**2 / 2 / mass / _m0
    #     return k.sum(0)

    # def pot(t, x0, pseudo):
    #     # TODO: Coulomb potential energy missing
    #     www = waveform_t(t / _t0)
    #     ww_index = 'tv' if n_samples > 1 else 'v'
    #     u = np.einsum(f'{ww_index},vnt->nt', www, trap.dc_potentials(x0, 0, trap.z0))
    #     if pseudo:
    #         u += trap.pseudo_potential(x0, 0, trap.z0)
    #     return _q0 * charge * u.sum(0)

    # x, p = sol.y[:N], sol.y[N:]  # these have shape (N, t)
    # sol.energy = kin(p) + pot(sol.t, x, pseudo=pseudo)

    return sol
