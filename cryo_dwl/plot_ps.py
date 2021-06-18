#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np
import matplotlib.pyplot as plt
from pytrans.trap_model.cryo import CryoTrap

trap = CryoTrap()
x = trap.transport_axis

plt.plot(x, trap.pseudo_potential(x))
plt.figure()
plt.plot(x, trap.pseudo_gradient(x).T)
plt.figure()
plt.imshow(trap.pseudo_hessian(0))



plt.show()