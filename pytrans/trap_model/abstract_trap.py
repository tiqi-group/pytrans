#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Module docstring

"""
from abc import ABC, abstractmethod


class AbstractTrap(ABC):

    @abstractmethod
    def load_trap_axis_potential_data(self):
        """I want that this defines:
        transport_axis: x
        moments: array (num_electrodes, len(x)) with electrode moments
        """
        pass
