#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''
from abc import ABC, abstractmethod
from typing import Union, List
from .trap_model.abstract_trap import AbstractTrap
from .potential_well import PotentialWell
import numpy as np
import cvxpy as cx
import operator

constraint_operator_mapping = {
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
    value: np.typing.ArrayLike = 0
    constraint_type = None

    @abstractmethod
    def objective(self, trap: AbstractTrap, voltages: cx.Variable, electrode_indices: Union[slice, List[int]] = slice(None)):
        """objective"""
        return
        yield

    def constraint(self, trap: AbstractTrap, voltages: cx.Variable):
        """constrain"""
        print(voltages, self.constraint_type, self.value)
        try:
            yield constraint_operator_mapping[self.constraint_type](voltages, self.value)  # type: ignore
        except KeyError:
            raise KeyError(f"Wrong constraint type defined: {self.constraint_type}")


class VoltageObjective(Objective):

    def __init__(self, value, weight=1., constraint_type=None):
        self.value = value
        self.weight = weight
        self.constraint_type = constraint_type

    def objective(self, trap, voltages, electrode_indices=slice(None)):
        cost = cx.multiply(self.weight, cx.sum_squares(voltages - self.value))
        yield cost


class CurvatureObjective(Objective):

    def __init__(self, x0, value, weight=1.):
        self.x0 = x0
        self.value = value
        self.weight = weight

    def objective(self, trap, voltages, electrode_indices=slice(None)):
        # assert len(electrode_indices) == voltages.shape[1], 'Wrong electrode indexing'
        xi = np.argmin(abs(self.x0 - trap.x))
        curvature = voltages @ trap.hessians[electrode_indices, 0, xi]
        cost = cx.multiply(self.weight, cx.sum_squares(curvature - self.value))
        yield cost


class PotentialWellObjective(Objective):

    def __init__(self, well: PotentialWell, ignore_offset=True):
        self.well = well
        self.extra_offset = cx.Variable((1,)) if ignore_offset else None
        
    def objective(self, trap, voltages, electrode_indices=slice(None)):
        roi = self.well.roi(trap.x)
        x = trap.x[roi]
        weight = self.well.weight(x)
        value = self.well.potential(x)
        if self.extra_offset:
            value = value + self.extra_offset
        print(voltages, np.where(roi)[0].shape)
        moments = trap.moments[electrode_indices][:, roi]
        print(voltages.shape, moments.shape)
        diff = (voltages @ moments - value)
        yield cx.sum_squares(cx.multiply(np.sqrt(weight), diff))
