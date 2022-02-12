#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''
from abc import ABC, abstractmethod
from typing import Union
from .trap_model.abstract_trap import AbstractTrap
from .potential_well import PotentialWell, MultiplePotentialWell
from .indexing import parse_indexing
import numpy as np
import cvxpy as cx
import operator

_constraint_operator_map = {
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '>=': operator.ge,
    '>': operator.gt,
    'lt': operator.lt,
    'le': operator.le,
    'eq': operator.eq,
    'ge': operator.ge,
    'gt': operator.gt,
}


class Objective(ABC):
    weight: float = 1.
    constraint_type = None

    def __init__(self, weight=1., constraint_type=None):
        self.weight = weight
        self.constraint_type = constraint_type

    @abstractmethod
    def objective(self, trap: AbstractTrap, voltages: cx.Variable):
        """objective"""
        return
        yield

    @abstractmethod
    def constraint(self, trap: AbstractTrap, voltages: cx.Variable):
        """constrain"""
        return
        yield

    def _yield_constraint(self, lhs, rhs):
        try:
            yield _constraint_operator_map[self.constraint_type](lhs, rhs)  # type: ignore
        except KeyError:
            raise KeyError(f"Wrong constraint type defined: {self.constraint_type}")


class VoltageObjective(Objective):

    def __init__(self, value, index=None, voltage_weights=None, weight=1., constraint_type=None):
        super().__init__(weight, constraint_type)
        self.value = value
        self.index = parse_indexing(index) if index is not None else index
        self.voltage_weights = voltage_weights

    def objective(self, trap, voltages):
        if self.index is not None:
            voltages = voltages[self.index]
        diff = voltages - self.value
        if self.voltage_weights is not None:
            diff = cx.multiply(np.sqrt(self.voltage_weights), diff)
        cost = cx.multiply(self.weight, cx.sum_squares(diff))
        yield cost

    def constraint(self, trap, voltages):
        if self.index is not None:
            voltages = voltages[self.index]
        return self._yield_constraint(voltages, self.value)


class SlewRateObjective(Objective):

    def __init__(self, weight=1., constraint_type=None):
        super().__init__(weight, constraint_type)

    def objective(self, trap, voltages):
        n_samples = voltages.shape[0]
        M = np.zeros((n_samples, n_samples))
        M += np.diagflat([1] * (n_samples - 1), k=1) / 2
        M += np.diagflat([-1] * (n_samples - 1), k=-1) / 2
        M[0, [0, 1]] = -1, 1
        M[-1, [-2, -1]] = -1, 1
        # print("--- grad")
        yield cx.multiply(self.weight, cx.sum_squares(M @ voltages))

    def constraint(self, trap, voltages):
        raise NotImplementedError


class SymmetryObjective(Objective):

    def __init__(self, lhs_indices, rhs_indices, **kwargs):
        super().__init__(**kwargs)
        self.lhs_indices = parse_indexing(lhs_indices)
        self.rhs_indices = parse_indexing(rhs_indices)

    def objective(self, trap, voltages):
        return
        yield

    def constraint(self, trap, voltages):
        return self._yield_constraint(voltages[self.lhs_indices], voltages[self.rhs_indices])


class PotentialObjective(Objective):

    def __init__(self, x0, derivatives, value, pseudo=True, **kwargs):
        super().__init__(**kwargs)
        self.x0 = x0
        self.derivatives = derivatives
        self.value = value
        self.pseudo = pseudo

    def objective(self, trap, voltages):
        # assert len(electrode_indices) == voltages.shape[1], 'Wrong electrode indexing'
        xi = np.argmin(abs(self.x0 - trap.x))
        pot = voltages @ trap.dc_potential(self.derivatives)[..., xi]
        if self.pseudo:
            pot += trap.pseudo_potential(self.derivatives)[..., xi]
        cost = cx.multiply(self.weight, cx.sum_squares(pot - self.value))
        yield cost

    def constraint(self, trap, voltages):
        # assert len(electrode_indices) == voltages.shape[1], 'Wrong electrode indexing'
        xi = np.argmin(abs(self.x0 - trap.x))
        pot = voltages @ trap.dc_potential(self.derivatives)[..., xi]
        if self.pseudo:
            pot += trap.pseudo_potential(self.derivatives)[..., xi]
        return self._yield_constraint(pot, self.value)


class GridPotentialObjective(Objective):

    def __init__(self, well: Union[PotentialWell, MultiplePotentialWell], optimize_offset=False, **kwargs):
        super().__init__(**kwargs)
        self.well = well
        self.extra_offset = cx.Variable((1,)) if optimize_offset else None

    def objective(self, trap, voltages):
        roi = self.well.roi(trap.x)
        x = trap.x[roi]
        weight = self.well.weight(x)
        value = self.well.potential(x)
        if self.extra_offset:
            value = value + self.extra_offset
        moments = trap.moments[..., roi]
        # v = voltages.value
        # if v is not None:
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots()
        #     ax.plot(trap.x * 1e6, v @ trap.moments[electrode_indices])
        #     ax.plot(x * 1e6, self.well.potential(x))
        #     plt.show()
        diff = (voltages @ moments - value)
        cost = cx.multiply(self.weight, cx.sum_squares(cx.multiply(np.sqrt(weight), diff)))
        yield cost

    def constraint(self, trap, voltages):
        raise NotImplementedError
