#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np
import cvxpy as cx


class Objective:

    weight: float = 1.


class PointPotentialObjective(Objective):

    def __init__(self, x):
        pass

    def objective(self, system, voltages):
        pass
