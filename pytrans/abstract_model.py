#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

import numpy as np
from typing import Union, List, Dict, Optional

from .electrode import DCElectrode, RFElectrode

import logging
logger = logging.getLogger(__name__)

ElectrodeNames = Union[str, List[str]]


class AbstractTrapModel:
    """This class does nothing, but it will force the implementation
    of some properties and methods in trap models that are used in the Objectives

    Required:
    - functions for 3d dc potentials and derivatives
    - functions for 3d RF pseudopotential and derivatives

    Optional?:
    - num_electrodes
    - electrode - channel mapping, for waveforms
    - electrode_constrains(min, max, slew rate)
    - filters?
    """
    __required_attributes = ['_dc_electrodes', '_rf_electrodes', '_rf_voltage', '_rf_freq_mhz']

    _dc_electrodes: Dict[str, DCElectrode]
    _rf_electrodes: Dict[str, RFElectrode]
    _all_electrodes: Dict[str, DCElectrode]
    _rf_voltage: float
    _rf_freq_mhz: float

    def __new__(cls, *args, **kwargs):
        for name in cls.__required_attributes:
            if not hasattr(cls, name):
                raise TypeError(f"Can't instantiate class {cls.__name__} without attribute {name}. Required attributes: {', '.join(cls.__required_attributes)}")

        cls._all_electrodes = cls._dc_electrodes
        return super().__new__(cls)

    def __init__(self, use_electrodes: Optional[List[str]] = None):
        if use_electrodes is not None:
            try:
                self._electrodes = {name: self._all_electrodes[name] for name in use_electrodes}
            except KeyError as e:
                raise KeyError(f"Trap {self.__class__.__name__} has no electrode {e}. Available electrodes: {list(self._all_electrodes.keys())}") from e
        else:
            self._electrodes = self._all_electrodes

    @property
    def electrodes(self) -> List[str]:
        """Names of DC electrodes in use"""
        return list(self._electrodes.keys())

    @property
    def rf_electrodes(self) -> List[str]:
        """Names of RF electrodes"""
        return list(self._rf_electrodes.keys())

    @property
    def n_electrodes(self):
        """Number of active electrodes
        """
        return len(self._electrodes)

    @classmethod
    def get_electrode(cls, name: str) -> DCElectrode:
        """Get DC electrode by name"""
        return cls._all_electrodes[name]

    @classmethod
    def get_rf_electrode(cls, name: str) -> RFElectrode:
        """Get RF electrode by name"""
        return cls._rf_electrodes[name]

    def dc_potentials(self, x, y, z):
        """
        Returns:
            out: ndarray, shape: (n_dc,) + x.shape
        """
        return np.stack([self.get_electrode(name).potential(x, y, z) for name in self.electrodes], axis=0)

    def dc_gradients(self, x, y, z):
        """
        Returns:
            out: ndarray, shape: (n_dc,) + x.shape + (3,)
        """
        # return np.stack([self.get_electrode(name).gradient(x, y, z) for name in self.electrodes], axis=0)
        return np.stack([self.get_electrode(name).gradient(x, y, z) for name in self.electrodes], axis=0)

    def dc_hessians(self, x, y, z):
        """
        Returns:
            out: ndarray, shape: (n_dc,) + x.shape + (3, 3)
        """
        return np.stack([self.get_electrode(name).hessian(x, y, z) for name in self.electrodes], axis=0)

    def pseudo_potential(self, x, y, z, ion_mass_amu):
        """Pseudopotential from RF

        This assumes all RF electrodes in phase and at the same RF voltage and frequency

        Returns:
            out: ndarray, shape: x.shape

        TODO In these conditions, it probably doesn't make sense to have
        more than one RFElectrode object in the model. Think about simplifying this.
        """
        return np.sum([rf_el.potential(x, y, z, ion_mass_amu, self._rf_voltage, self._rf_freq_mhz)
                       for rf_el in self._rf_electrodes.values()], axis=0)

    def pseudo_gradient(self, x, y, z, ion_mass_amu):
        """Pseudopotential gradient

        Returns:
            out: ndarray, shape: x.shape + (3,)
        """
        return np.sum([rf_el.gradient(x, y, z, ion_mass_amu, self._rf_voltage, self._rf_freq_mhz)
                       for rf_el in self._rf_electrodes.values()], axis=0)

    def pseudo_hessian(self, x, y, z, ion_mass_amu):
        """Pseudopotential curvatures

        Returns:
            out: ndarray, shape: x.shape + (3, 3)
        """
        return np.sum([rf_el.hessian(x, y, z, ion_mass_amu, self._rf_voltage, self._rf_freq_mhz)
                       for rf_el in self._rf_electrodes.values()], axis=0)

    def potential(self, voltages, x, y, z, ion_mass_amu, pseudo=True):
        u = np.tensordot(voltages, self.dc_potentials(x, y, z), axes=1)
        if len(self._rf_electrodes) > 0 and pseudo:
            u += self.pseudo_potential(x, y, z, ion_mass_amu)
        return u

    def gradient(self, voltages, x, y, z, ion_mass_amu, pseudo=True):
        u = np.tensordot(voltages, self.dc_gradients(x, y, z), axes=1)
        if len(self._rf_electrodes) > 0 and pseudo:
            u += self.pseudo_gradient(x, y, z, ion_mass_amu)
        return u

    def hessian(self, voltages, x, y, z, ion_mass_amu, pseudo=True):
        u = np.tensordot(voltages, self.dc_hessians(x, y, z), axes=1)
        if len(self._rf_electrodes) > 0 and pseudo:
            u += self.pseudo_hessian(x, y, z, ion_mass_amu)
        return u

    @property
    def electrode_all_indices(self):
        return self.electrode_to_index(self.electrodes, in_all=True)

    def electrode_to_index(self, names: ElectrodeNames, in_all=False):
        el_list = list(self._all_electrodes.keys()) if in_all else self.electrodes
        if isinstance(names, str):
            return el_list.index(names)
        return [el_list.index(n) for n in names]

    def fill_waveform(self, waveform):
        waveform = np.atleast_2d(waveform)
        waveform0 = np.zeros((len(waveform), len(self._all_electrodes)))
        waveform0[:, self.electrode_all_indices] = waveform
        return waveform0
