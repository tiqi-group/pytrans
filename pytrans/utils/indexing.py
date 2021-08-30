#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Inspired by https://github.com/nist-ionstorage/electrode/blob/master/electrode/utils.py
'''

from collections import defaultdict

_derivative_names = [[""]] + [s.split() for s in [
    "x y z",
    "xx xy xz yy yz zz",
]]

_name_to_index_map = {}
_order_to_index_map = defaultdict(list)


def _populate_maps():
    idx = 0
    for order, names in enumerate(_derivative_names):
        for name in names:
            _name_to_index_map[name] = idx
            _order_to_index_map[order].append(idx)
            idx += 1


_populate_maps()


def get_derivative(d):
    if isinstance(d, int):
        return _order_to_index_map[d]
    elif isinstance(d, str):
        return _name_to_index_map[d]
    elif isinstance(d, list):
        return [_name_to_index_map[name] for name in d]
    else:
        raise TypeError(f"Undefined derivative: {d}")


if __name__ == '__main__':
    print(_derivative_names)
    __import__('pprint').pprint(_name_to_index_map)
    __import__('pprint').pprint(_order_to_index_map)
