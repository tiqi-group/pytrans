#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 11/2023
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

import sys
import pytrans  # noqa
from pytrans import analysis  # noqa
from pytrans import plotting  # noqa


def test_import():
    assert "pytrans" in sys.modules
