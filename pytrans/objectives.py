#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''
from abc import ABC, abstractmethod
from .abstract_model import AbstractTrap
from .indexing import parse_indexing, get_derivative, gradient_matrix
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
        M = gradient_matrix(n_samples)
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
        raise NotImplementedError

    def constraint(self, trap, voltages):
        return self._yield_constraint(voltages[self.lhs_indices], voltages[self.rhs_indices])


class PotentialObjective(Objective):

    def __init__(self, x, y, z, value, pseudo=True, weight=1., constraint_type=None):
        super().__init__(weight, constraint_type)
        self.xyz = x, y, z
        self.value = value
        self.pseudo = pseudo

    def objective(self, trap, voltages):
        pot = voltages @ trap.dc_potentials(*self.xyz)
        if self.pseudo:
            pot += trap.pseudo_potential(*self.xyz)
        cost = cx.multiply(self.weight, cx.sum_squares(pot - self.value))
        yield cost

    def constraint(self, trap, voltages):
        pot = voltages @ trap.dc_potentials(*self.xyz)
        if self.pseudo:
            pot += trap.pseudo_potential(*self.xyz)
        return self._yield_constraint(pot, self.value)


class GradientObjective(Objective):

    def __init__(self, x, y, z, value, entries=None, pseudo=True, weight=1., constraint_type=None):
        super().__init__(weight, constraint_type)
        self.xyz = x, y, z
        self.value = value
        self.pseudo = pseudo
        self.entries = slice(None) if entries is None else get_derivative(entries)

    def objective(self, trap, voltages):
        pot = voltages @ trap.dc_gradients(*self.xyz)  # resulting shape is (3, len(x))
        if self.pseudo:
            pot += trap.pseudo_gradient(*self.xyz)
        pot = pot[self.entries]
        cost = cx.multiply(self.weight, cx.sum_squares(pot - self.value))
        yield cost

    def constraint(self, trap, voltages):
        pot = voltages @ trap.dc_gradients(*self.xyz)
        if self.pseudo:
            pot += trap.pseudo_gradient(*self.xyz)
        pot = pot[self.entries]
        return self._yield_constraint(pot, self.value)


class HessianObjective(Objective):

    def __init__(self, x, y, z, value, entries=None, pseudo=True, weight=1., constraint_type=None):
        super().__init__(weight, constraint_type)
        self.xyz = x, y, z
        self.value = value
        self.pseudo = pseudo
        self.entries = slice(None) if entries is None else get_derivative(entries)

    def objective(self, trap, voltages):
        nv = voltages.shape[-1]
        pot = voltages @ trap.dc_hessians(*self.xyz).reshape(nv, 9)
        if self.pseudo:
            pot += trap.pseudo_hessian(*self.xyz).reshape(9)
        pot = pot[self.entries]
        cost = cx.multiply(self.weight, cx.sum_squares(pot - self.value))
        yield cost

    def constraint(self, trap, voltages):
        nv = voltages.shape[-1]
        pot = voltages @ trap.dc_hessians(*self.xyz).reshape(nv, 9)
        if self.pseudo:
            pot += trap.pseudo_hessian(*self.xyz).reshape(9)
        pot = pot[self.entries]
        return self._yield_constraint(pot, self.value)
