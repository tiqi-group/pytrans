#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 03/2023
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>
""" Define common types
"""
from typing import Union, List
from nptyping import NDArray, Shape, Float

ElectrodeNames = Union[str, List[str]]
Coords = NDArray[Shape["*, 3"], Float]
