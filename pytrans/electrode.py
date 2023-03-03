#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 03/2023
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>


from typing import Any
from abc import ABC, abstractmethod
from nptyping import NDArray, Float


class Electrode(ABC):
    """TODO implement factory method

    It would be nice to have the alternatives:
    - subclass DCElectrode or RFElectrode and implement methods yourself
    - use a factory method of the class
    """

    @abstractmethod
    def _unit_potential(self, x: NDArray, y: NDArray, z: NDArray) -> NDArray[Any, Float]:
        """
        Args:
            x, y, z (array_like): 3D coordinates. Must be broadcastable.

        Returns:
            out (ndarray, shape: x.shape)
        """
        # return self._unit_potential_func(x, y, z)
        raise NotImplementedError

    @abstractmethod
    def _unit_gradient(self, x: NDArray, y: NDArray, z: NDArray) -> NDArray[Any, Float]:
        """
        Args:
            x, y, z (array_like): 3D coordinates. Must be broadcastable.

        Returns:
            out (ndarray, shape: x.shape + (3,))
        """
        raise NotImplementedError

    @abstractmethod
    def _unit_hessian(self, x: NDArray, y: NDArray, z: NDArray) -> NDArray[Any, Float]:
        """
        Args:
            x, y, z (array_like): 3D coordinates. Must be broadcastable.

        Returns:
            out (ndarray, shape: x.shape + (3, 3))
        """
        raise NotImplementedError


class DCElectrode(Electrode):

    def potential(self, x, y, z):
        return self._unit_potential(x, y, z)

    def gradient(self, x, y, z):
        return self._unit_gradient(x, y, z)

    def hessian(self, x, y, z):
        return self._unit_hessian(x, y, z)


class RFElectrode(Electrode):

    def kappa(self, ion_mass_amu, rf_voltage, rf_freq_mhz):
        return rf_voltage**2 / ion_mass_amu / rf_freq_mhz**2

    def potential(self, x, y, z, ion_mass_amu, rf_voltage, rf_freq_mhz):
        return self.kappa(ion_mass_amu, rf_voltage, rf_freq_mhz) * self._unit_potential(x, y, z)

    def gradient(self, x, y, z, ion_mass_amu, rf_voltage, rf_freq_mhz):
        return self.kappa(ion_mass_amu, rf_voltage, rf_freq_mhz) * self._unit_gradient(x, y, z)

    def hessian(self, x, y, z, ion_mass_amu, rf_voltage, rf_freq_mhz):
        return self.kappa(ion_mass_amu, rf_voltage, rf_freq_mhz) * self._unit_hessian(x, y, z)
