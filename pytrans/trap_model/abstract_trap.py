#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Module docstring

"""
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

    @property
    @abstractmethod
    def moments(self) -> ArrayLike:
        '''trap moments: shape (n_electrodes, len(x))'''
        pass
