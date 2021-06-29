#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

from pytrans.trap_model.cryo import CryoTrap, hessiansDC

trap = CryoTrap()

print(trap.moments.shape)
print(trap.gradients.shape)
print(trap.hessians.shape)

h1 = getattr(hessiansDC, "E1")(trap.transport_axis[0], 0, trap.z0)
print(h1.shape)
print(h1)
print(trap.hessians[0, :, 0])
