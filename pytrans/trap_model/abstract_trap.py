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
