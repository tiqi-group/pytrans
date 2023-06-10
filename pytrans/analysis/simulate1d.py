#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 05/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from tqdm import tqdm
from pytrans.timer import timer


# length and time scales
from scipy.constants import atomic_mass as _m0
from scipy.constants import elementary_charge as _q0
from scipy.constants import epsilon_0
_x0 = 1e-6
_t0 = 1e-6

_p0 = _m0 * _x0 / _t0
_E0 = _m0 * _x0 / _q0 / _t0**2
_K0 = 1 / 4 / np.pi / epsilon_0 * _q0**2 * _t0**2 / _m0 / _x0**3

solve_ivp = timer(solve_ivp)


def coulomb_force_a(x1, x2):
    return _K0 / (x2 - x1)**2 * np.asarray([1, -1])


def simulate_waveform(trap, waveform, t, x0, bounds=None, slowdown=1, pseudo=False, solve_kw=dict()):

    mass = trap.ion.mass / _m0
    charge = trap.ion.charge / _q0
    x0 = np.asarray(x0)

    n_samples = len(waveform)
    if n_samples == 1:
        def waveform_t(t):
            return waveform[0]
    else:
        s = np.linspace(0, 1, n_samples)
        waveform_s = interp1d(s, waveform, axis=0,
                              bounds_error=False, fill_value=(waveform[0], waveform[-1]))

        def waveform_t(t):
            return waveform_s(t * _t0 / (trap.dt * n_samples * slowdown))

    n_ions = len(x0)
    assert n_ions in [1, 2]
    print(n_ions)

    y0 = np.r_[x0 / _x0, [0] * n_ions]

    def trap_force(t, x0):
        return - charge * trap.gradient(waveform_t(t), x0 * _x0, 0, trap.z0, pseudo=pseudo)[0] / _E0

    if n_ions > 1:
        def force(t, x0):
            return - charge**2 * coulomb_force_a(*x0) + trap_force(t, x0)
    else:
        force = trap_force

    if bounds is not None:
        assert isinstance(bounds, (list, tuple)) and len(bounds) == 2
        events = []
        for j in range(n_ions):
            def exit_event(t, y, *args):
                return (y[j] - bounds[0] / _x0) * (y[j] - bounds[1] / _x0)
            exit_event.terminal = True
            events.append(exit_event)
    else:
        events = None

    def fun(t, y):
        # y = x1, x2, p1, p2
        return np.r_[y[n_ions:] / mass, force(t, y[:n_ions])]

    def fun_pbar(t, y, pbar, state):
        # https://stackoverflow.com/a/62140877
        last_t, dt = state
        n = int((t - last_t) / dt)
        pbar.update(n)
        state[0] = last_t + dt * n
        return fun(t, y)

    t0, t1 = t[[0, -1]] / _t0
    state = [t0, (t1 - t0) / 1000]

    kw = dict(t_eval=t / _t0, dense_output=True, events=events, method='LSODA')
    kw.update(solve_kw)

    with tqdm(total=1000, unit="%") as pbar:
        sol = solve_ivp(fun_pbar, (t0, t1), y0, args=(pbar, state), **kw)

    sol.t = sol.t * _t0
    sol.y = sol.y * np.r_[[_x0] * n_ions, [_p0] * n_ions].reshape(-1, 1)

    def kin(p):
        k = p**2 / 2 / mass / _m0
        return k.sum(0)

    def pot(t, x0, pseudo):
        www = waveform_t(t / _t0)
        ww_index = 'tv' if n_samples > 1 else 'v'
        u = np.einsum(f'{ww_index},vnt->nt', www, trap.dc_potentials(x0, 0, trap.z0))
        if pseudo:
            u += trap.pseudo_potential(x0, 0, trap.z0)
        return _q0 * charge * u.sum(0)

    x, p = sol.y[:n_ions], sol.y[n_ions:]  # these have shape (n_ions, t)
    sol.energy = kin(p) + pot(sol.t, x, pseudo=pseudo)

    return sol
