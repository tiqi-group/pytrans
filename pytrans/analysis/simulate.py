#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 05/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from tqdm import tqdm

from pytrans.analysis import coulomb as pcoul


def simulate_waveform(trap, waveform, t, x0, bounds=None, slowdown=1, solve_kw=dict()):
    n_samples = len(waveform)
    s = np.linspace(0, 1, n_samples)
    waveform_s = interp1d(s, waveform, axis=0,
                          bounds_error=False, fill_value=(waveform[0], waveform[-1]))

    mass = trap.ion.mass
    charge = trap.ion.charge
    dt = trap.dt

    n_ions = len(x0)
    assert n_ions in [1, 2]
    print(n_ions)

    y0 = np.r_[x0, [0] * n_ions]

    if n_ions > 1:
        def grad(t, x0):
            return trap.gradient(waveform_s(t / (dt * n_samples * slowdown)),
                                 x0, 0, trap.z0, pseudo=False)[0] + charge * pcoul.coulomb_grad(*x0)
    else:
        def grad(t, x0):
            return trap.gradient(waveform_s(t / (dt * n_samples * slowdown)),
                                 x0, 0, trap.z0, pseudo=False)[0]

    if bounds is not None:
        assert isinstance(bounds, (list, tuple)) and len(bounds) == 2
        events = []
        for j in range(n_ions):
            def exit_event(t, y, *args):
                return (y[j] - bounds[0]) * (y[j] - bounds[1])
            exit_event.terminal = True
            events.append(exit_event)
    else:
        events = None

    def fun(t, y):
        # y = x1, x2, p1, p2
        return np.r_[y[n_ions:] / mass, - charge * grad(t, y[:n_ions])]

    def fun_pbar(t, y, pbar, state):
        # https://stackoverflow.com/a/62140877
        last_t, dt = state
        n = int((t - last_t) / dt)
        pbar.update(n)
        state[0] = last_t + dt * n
        return fun(t, y)

    t0, t1 = t[[0, -1]]
    state = [t0, (t1 - t0) / 1000]

    kw = dict(t_eval=t, dense_output=True, events=events, method='LSODA')
    kw.update(solve_kw)

    with tqdm(total=1000, unit="%") as pbar:
        sol = solve_ivp(fun_pbar, (t0, t1), y0, args=(pbar, state), **kw)

    return sol
