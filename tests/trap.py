#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

import pytest
from pytrans.trap_model.test_trap import TestTrap

TestTrap.__test__ = False


@pytest.fixture
def trap() -> TestTrap:
    return TestTrap()
