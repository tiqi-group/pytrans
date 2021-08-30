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
from .potential_well import PotentialWell, MultiplePotentialWell
import numpy as np
import cvxpy as cx
import operator

constraint_operator_map = {
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
    value: np.typing.ArrayLike = 0
    weight: float = 1.
    constraint_type = None

    def __init__(self, weight=1., constraint_type=None):
        self.weight = weight
        self.constraint_type = constraint_type

    @abstractmethod
    def objective(self, trap: AbstractTrap, voltages: cx.Variable, electrode_indices: Union[slice, List[int]] = slice(None)):
        """objective"""
        return
        yield

    def constraint(self, trap: AbstractTrap, voltages: cx.Variable, electrode_indices: Union[slice, List[int]] = slice(None)):
        """constrain"""
        return
        yield


class VoltageObjective(Objective):

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def objective(self, trap, voltages, electrode_indices=slice(None)):
        cost = cx.multiply(self.weight, cx.sum_squares(voltages - self.value))
        yield cost

    def constraint(self, trap, voltages, electrode_indices):
        try:
            yield constraint_operator_map[self.constraint_type](voltages, self.value)  # type: ignore
        except KeyError:
            raise KeyError(f"Wrong constraint type defined: {self.constraint_type}")


class PotentialObjective(Objective):

    def __init__(self, x0, derivatives, value, **kwargs):
        super().__init__(**kwargs)
        self.x0 = x0
        self.derivatives = derivatives
        self.value = value

    def objective(self, trap, voltages, electrode_indices=slice(None)):
        # assert len(electrode_indices) == voltages.shape[1], 'Wrong electrode indexing'
        xi = np.argmin(abs(self.x0 - trap.x))
        pot = voltages @ trap.dc_potential(self.derivatives, electrode_indices)[..., xi] + trap.pseudo_potential(self.derivatives)[..., xi]
        v = voltages.value
        if v is not None:
            vpot = v @ trap.dc_potential(self.derivatives, electrode_indices)[..., xi] + trap.pseudo_potential(self.derivatives)[..., xi]
            print(vpot.shape, vpot)
        cost = cx.multiply(self.weight, cx.sum_squares(pot - self.value))
        yield cost

    def constraint(self, trap, voltages, electrode_indices=slice(None)):
        # assert len(electrode_indices) == voltages.shape[1], 'Wrong electrode indexing'
        xi = np.argmin(abs(self.x0 - trap.x))
        pot = voltages @ trap.dc_potential(self.derivatives, electrode_indices)[..., xi] + trap.pseudo_potential(self.derivatives)[..., xi]
        try:
            yield constraint_operator_map[self.constraint_type](pot, self.value)  # type: ignore
        except KeyError:
            raise KeyError(f"Wrong constraint type defined: {self.constraint_type}")


class GridPotentialObjective(Objective):

    def __init__(self, well: Union[PotentialWell, MultiplePotentialWell], optimize_offset=False, **kwargs):
        super().__init__(**kwargs)
        self.well = well
        self.extra_offset = cx.Variable((1,)) if optimize_offset else None

    def objective(self, trap, voltages, electrode_indices=slice(None)):
        roi = self.well.roi(trap.x)
        x = trap.x[roi]
        weight = self.well.weight(x)
        value = self.well.potential(x)
        if self.extra_offset:
            value = value + self.extra_offset
        moments = trap.moments[electrode_indices][..., roi]
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

    def constraint(self, trap, voltages, electrode_indices):
        raise NotImplementedError
