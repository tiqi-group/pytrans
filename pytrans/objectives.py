#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

r"""
Objectives module

probably worth defining the symbols here, or in another MD file

- Voltages $v_i$
- moments $\phi_i = \phi_i(x, y, z)$

"""
from abc import ABC, abstractmethod
import operator
from typing import Union, List
import numpy as np
import cvxpy as cx
from numpy.typing import ArrayLike
from .abstract_model import AbstractTrap
from .indexing import get_derivative, gradient_matrix

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
    """Objective
    A building block of the cost function.
    CLasses inheriting from `Objective` can be used to generate specific terms of the
    cost function to be optimized, or to implement constraints on the optimization
    variables.
    """
    weight = 1.0
    constraint_type = None

    def __init__(self, weight: float = 1.0, constraint_type: str = None):
        """Initialize abstract Objective class.
            The `constraint_type` attribute defines wether the class will be used
            as cost term or a constraint. If `None`, no constraint is implemented
            and the Objective is used to generate a term of the cost function.
            Otherwise, it must be one of
            `['<', '<=', '==', '>=', '>', 'lt','le', 'eq', 'ge', 'gt']`
            and will implement the corresponding constraint on the optimization variables.

        Args:
            weight (float, optional): global weight of the cost term.
            constraint_type (str, optional): a string defining the constraint type.
        """
        self.weight = weight
        self.constraint_type = constraint_type

    @abstractmethod
    def objective(self, trap: AbstractTrap, voltages: cx.Variable):
        """override this method to implement a cost objective

        Args:
            trap (AbstractTrap): trap model
            voltages (cx.Variable): optimization variables
        """
        return
        yield

    @abstractmethod
    def constraint(self, trap: AbstractTrap, voltages: cx.Variable):
        """override this method to implement a constraint

        Args:
            trap (AbstractTrap): trap model
            voltages (cx.Variable): optimization variables
        """
        return
        yield

    def _yield_constraint(self, lhs, rhs):
        try:
            yield _constraint_operator_map[self.constraint_type](lhs, rhs)
        except KeyError as e:
            raise KeyError(
                f"Wrong constraint type defined: {self.constraint_type}") from e


class VoltageObjective(Objective):

    def __init__(self, value: ArrayLike, electrodes: Union[str, List[str]] = None,
                 local_weights: ArrayLike = None,
                 weight: float = 1.0, constraint_type: str = None):
        r"""Implements an Objective for the voltages applied to the trap electrodes

        Cost
            $$ \mathcal{C} = w\, \sum_{i \in \mathcal{E}} w_i\left( v_i - x_i \right)^2. $$

        Constraint
            $$ v_i \leq x_i \quad \forall i \in \mathcal{E}. $$

        Args:
            value (ArrayLike): $x_i$
                target voltage for the selected set of electrodes
            electrodes (string, or list of strings, optional): $\mathcal{E}$
                specify a subset of the trap electrodes. If `None`, uses all electrodes.
            local_weights (ArrayLike, optional): $w_i$
                per-electrode weight. If `None`, all weights are set to 1.
            weight (float, optional): $w$
                global weight of the cost term.
            constraint_type (str, optional):
                constraint string.

        """
        super().__init__(weight, constraint_type)
        self.value = value
        self.electrodes = electrodes
        self.local_weights = local_weights

    def objective(self, trap: AbstractTrap, voltages: cx.Variable):
        """objective"""
        if self.electrodes is not None:
            electrodes = trap.electrode_to_index(self.electrodes, in_all=False)
            voltages = voltages[:,electrodes]
        diff = voltages - self.value
        if self.local_weights is not None:
            diff = cx.multiply(np.sqrt(self.local_weights), diff)
        cost = cx.multiply(self.weight, cx.sum_squares(diff))
        yield cost

    def constraint(self, trap, voltages):
        """constraint"""
        if self.electrodes is not None:
            electrodes = trap.electrode_to_index(self.electrodes)
            voltages = voltages[:,electrodes]
        return self._yield_constraint(voltages, self.value)


class SlewRateObjective(Objective):

    def __init__(self, weight: float = 1.0):
        r"""Implements a cost penalizing the time derivative of the waveform.
            As such, it must be used as a global objective.

        Cost
            $$ \mathcal{C} = w\, \sum_{ji}\left( M_{jt} v_{ti} \right)^2, $$

        where $M_{ij}$ implements the second-order accurate finite difference
            $$ M_{ij}v_j = \frac{ v\_{i + 1} - v\_{i - 1} }{ 2 } $$

        (see e.g. `np.gradient`).

        No constraint is implemented.

        Args:
            weight (float, optional): $w$
                global weight of the cost term.
        """
        super().__init__(weight, constraint_type=None)

    def objective(self, trap, voltages):
        """objective"""
        if len(voltages.shape) < 2:
            raise ValueError(f"Not a global objective: wrong waveform shape ({voltages.shape})")
        n_samples = voltages.shape[0]
        M = gradient_matrix(n_samples)
        yield cx.multiply(self.weight, cx.sum_squares(M @ voltages))

    def constraint(self, trap, voltages):
        raise NotImplementedError


class SymmetryObjective(Objective):

    def __init__(self, lhs_indices: Union[str, List[str]],
                 rhs_indices: Union[str, List[str]],
                 sign: float = 1.0,
                 weight: float = 1.0, constraint_type: str = None):
        super().__init__(weight, constraint_type)
        self.lhs_indices = lhs_indices
        self.rhs_indices = rhs_indices
        self.sign = sign

    def objective(self, trap, voltages):
        raise NotImplementedError

    def constraint(self, trap, voltages):
        lhs = trap.electrode_to_index(self.lhs_indices)
        rhs = trap.electrode_to_index(self.rhs_indices)
        return self._yield_constraint(voltages[lhs], self.sign * voltages[rhs])

class UnusedElectrodesConstraint(Objective):
    """Implements a constraint to force unused electrodes to zero voltage"""
    def __init__(self, indices: Union[str, List[str]],
                 weight: float = 1.0, constraint_type: str = None):
        super().__init__(weight, constraint_type)
        self.indices = indices

    def objective(self, trap, voltages):
        raise NotImplementedError

    def constraint(self, trap, voltages):
        unused = trap.electrode_to_index(self.indices)
        return self._yield_constraint(voltages[unused], 0*voltages[unused])

class PotentialObjective(Objective):

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, value: ArrayLike,
                 pseudo: bool = True, local_weights: ArrayLike = None, norm: float = 1.0,
                 weight: float = 1.0, constraint_type: str = None):
        r"""Implements an Objective for the potential generated by the trap electrodes

        Cost
            $$ \mathcal{C} = w\, \sum_{x} w(x)\left(v_i \frac{\phi_i (x) - U(x)}{\mu} \right)^2. $$

        Constraint
            $$ v_i \leq x_i \quad \forall i \in \mathcal{E}. $$

        Args:
            x, y, z (ArrayLike): $x$
                coordinates where to evaluate the potential. Must be broadcastable.
            value (ArrayLike): $U(x)$
                target potential
            pseudo (bool, optional)
                select if to add the pseudopotential
            local_weights (ArrayLike, optional): $w(x)$
                position-dependent weight. If `None`, all weights are set to 1.
            norm (float, optional): $\mu$
                Charachteristic magnitude of the objective used to normalize the cost.
            weight (float, optional): $w$
                global weight of the cost term.
            constraint_type (str, optional):
                constraint string.

        """
        super().__init__(weight, constraint_type)
        self.xyz = x, y, z
        self.value = value
        self.pseudo = pseudo
        self.norm = norm
        self.local_weights = local_weights

    def objective(self, trap, voltages):
        """objective"""
        pot = voltages @ trap.dc_potentials(*self.xyz)
        if self.pseudo:
            pot += trap.pseudo_potential(*self.xyz)
        diff = (pot - self.value) / self.norm
        if self.local_weights is not None:
            diff = cx.multiply(np.sqrt(self.local_weights), diff)
        cost = cx.multiply(self.weight, cx.sum_squares(diff))
        yield cost

    def constraint(self, trap, voltages):
        """constraint"""
        pot = voltages @ trap.dc_potentials(*self.xyz)
        if self.pseudo:
            pot += trap.pseudo_potential(*self.xyz)
        return self._yield_constraint(pot, self.value)


class GradientObjective(Objective):

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, value: ArrayLike,
                 entries: Union[int, str, List[Union[int, str]]] = None,
                 pseudo: bool = True, norm: float = 1.0,
                 weight: float = 1.0, constraint_type: str = None):
        r"""Implements an Objective for the potential gradient generated by the trap electrodes

        Cost
            $$ \mathcal{C} = w\, \sum_{x} w(x)\left(v_i \frac{\partial_L \phi_i (x) - U(x)}{\mu} \right)^2. $$

        Constraint
            $$ v_i \leq x_i \quad \forall i \in \mathcal{E}. $$

        Args:
            x,y,z (ArrayLike): $x$
                coordinates where to evaluate the potential. Must be broadcastable.
            value (ArrayLike): $U(x)$
                target potential
            entries (str, optional) $L$
                index or string selecting the derivative of interest.\
                This needs to be explained better.
            pseudo (bool, optional)
                select if to add the pseudopotential
            local_weights (ArrayLike, optional): $w(x)$
                position-dependent weight. If `None`, all weights are set to 1.
            norm (float, optional): $\mu$
                Charachteristic magnitude of the objective used to normalize the cost.
            weight (float, optional): $w$
                global weight of the cost term.
            constraint_type (str, optional):
                constraint string.

        """
        super().__init__(weight, constraint_type)
        self.xyz = x, y, z
        self.value = value
        self.pseudo = pseudo
        self.norm = norm
        self.entries = slice(None) if entries is None else get_derivative(entries)

    def objective(self, trap, voltages):
        # resulting shape is (3, len(x))
        pot = voltages @ trap.dc_gradients(*self.xyz)
        if self.pseudo:
            pot += trap.pseudo_gradient(*self.xyz)
        pot = pot[self.entries]
        diff = (pot - self.value) / self.norm
        cost = cx.multiply(self.weight, cx.sum_squares(diff))
        yield cost

    def constraint(self, trap, voltages):
        pot = voltages @ trap.dc_gradients(*self.xyz)
        if self.pseudo:
            pot += trap.pseudo_gradient(*self.xyz)
        pot = pot[self.entries]
        return self._yield_constraint(pot, self.value)


class HessianObjective(Objective):

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, value: ArrayLike,
                 entries: Union[int, str, List[Union[int, str]]] = None,
                 pseudo: bool = True, norm: float = 1.0,
                 weight: float = 1.0, constraint_type: str = None):
        r"""Implements an Objective for the potential gradient generated by the trap electrodes

        Cost
            $$ \mathcal{C} = w\, \sum_{x} w(x)\left(v_i \frac{\partial^2_L \phi_i (x) - U(x)}{\mu} \right)^2. $$

        Constraint
            $$ v_i \leq x_i \quad \forall i \in \mathcal{E}. $$

        Args:
            x,y,z (ArrayLike): $x$
                coordinates where to evaluate the potential. Must be broadcastable.
            value (ArrayLike): $U(x)$
                target potential
            entries (str, optional) $L$
                index or string selecting the derivative of interest.\
                This needs to be explained better.
            pseudo (bool, optional)
                select if to add the pseudopotential
            local_weights (ArrayLike, optional): $w(x)$
                position-dependent weight. If `None`, all weights are set to 1.
            norm (float, optional): $\mu$
                Charachteristic magnitude of the objective used to normalize the cost.
            weight (float, optional): $w$
                global weight of the cost term.
            constraint_type (str, optional):
                constraint string.

        """
        super().__init__(weight, constraint_type)
        self.xyz = x, y, z
        self.value = value
        self.pseudo = pseudo
        self.norm = norm
        self.entries = slice(None) if entries is None else get_derivative(entries)

    def objective(self, trap, voltages):
        nv = voltages.shape[-1]
        pot = voltages @ trap.dc_hessians(*self.xyz).reshape(nv, 9)
        if self.pseudo:
            pot += trap.pseudo_hessian(*self.xyz).reshape(9)
        pot = pot[self.entries]
        diff = (pot - self.value) / self.norm
        cost = cx.multiply(self.weight, cx.sum_squares(diff))
        yield cost

    def constraint(self, trap, voltages):
        nv = voltages.shape[-1]
        pot = voltages @ trap.dc_hessians(*self.xyz).reshape(nv, 9)
        if self.pseudo:
            pot += trap.pseudo_hessian(*self.xyz).reshape(9)
        pot = pot[self.entries]
        return self._yield_constraint(pot, self.value)


class QuarticObjective(Objective):

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, value: ArrayLike,
                 norm: float = 1.0, weight: float = 1.0, constraint_type: str = None):
        super().__init__(weight, constraint_type)
        self.xyz = x, y, z
        self.value = value
        self.norm = norm

    def objective(self, trap, voltages):
        if not hasattr(trap, "dc_fourth_order"):
            raise NotImplementedError(
                "The current trap model does not implement fourth-order moments.")
        pot = voltages @ trap.dc_fourth_order(*self.xyz)
        if self.value == 'minimize':
            cost = cx.sum(pot / self.norm)
        elif self.value == 'maximize':
            cost = - cx.sum(pot / self.norm)
        else:
            cost = cx.sum_squares(pot - self.value / self.norm)
        cost = cx.multiply(self.weight, cost)
        yield cost

    def constraint(self, trap, voltages):
        if self.value in ["minimize", "maximize"]:
            raise NotImplementedError(
                f"Cannot {self.value} quartic term with a hard constraint.")
        pot = voltages @ trap.dc_fourth_order(*self.xyz)
        return self._yield_constraint(pot, self.value)
