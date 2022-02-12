#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

try:
    import gradientsDC_aot
    from ._dc import vectorize_stack
    moment0 = vectorize_stack(gradientsDC_aot.E6)
except ImportError:
    print("Python moment funcs for TestTrap")
    from ._dc import moment0

_el_width = 125e-6


def E1(x, y, z):
    return moment0(x + _el_width, y, z)


def E2(x, y, z):
    return moment0(x, y, z)


def E3(x, y, z):
    return moment0(x - _el_width, y, z)


def E4(x, y, z):
    return moment0(x + _el_width, -y, z)


def E5(x, y, z):
    return moment0(x, -y, z)


def E6(x, y, z):
    return moment0(x - _el_width, -y, z)
