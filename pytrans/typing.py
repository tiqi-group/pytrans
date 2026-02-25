#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 03/2023
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>
""" Define common types
"""
from typing import Union, List, Tuple
import numpy as np

ElectrodeNames = Union[str, List[str]]
Coords = np.ndarray
Coords1 = np.ndarray
RoiSize = Tuple[float]
Bounds = List[Tuple[float, float]]

Waveform = np.ndarray
