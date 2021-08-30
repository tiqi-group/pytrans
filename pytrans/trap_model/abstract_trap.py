#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Module docstring

"""
from typing import Union, List
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod

import logging
logger = logging.getLogger(__name__)


class AbstractTrap(ABC):
    """ I have to define here:
    - transport axis (alias to x)
    - moments and their derivatives
    - pseudo_potential and its derivatives
    - num_electrodes
    - electrode mapping
    - electrode_constrains (min, max, slew rate)
    - filters?
    """
    num_electrodes: int

    @property
    @abstractmethod
    def x(self) -> ArrayLike:
        '''transport axis'''
        pass

    @abstractmethod
    def dc_potential(self, derivatives: Union[int, str, List[str]], electrode_indices: Union[slice, List[int], ArrayLike]) -> ArrayLike:
        '''dc potentials: shape (n_electrodes, derivative, len(x))'''
        pass

    @abstractmethod
    def pseudo_potential(self, derivatives: Union[int, str, List[str]]) -> ArrayLike:
        '''pseudo potential: shape (derivative, len(x))'''
        pass
