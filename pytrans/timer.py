#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>

"""
Timing execution
https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
"""

import time
import logging

logger = logging.getLogger(__name__)


def timer(method):
    def timed(*args, **kwargs):
        print(f"Exec {method.__name__}")
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        elapsed = te - ts
        print(f"- {method.__name__} elapsed time: {elapsed * 1e3:.3f} ms")
        # logger.info(f"Exec {method.__name__} elapsed time: {elapsed * 1e3:.3f} ms")
        # if 'log_time' in kw:
        #     name = kw.get('log_name', method.__name__.upper())
        #     kw['log_time'][name] = int((te - ts) * 1000)
        # else:
        #     print '%r  %2.2f ms' % \
        #           (method.__name__, (te - ts) * 1000)
        return result

    return timed
