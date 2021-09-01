#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Inspired by https://github.com/nist-ionstorage/electrode/blob/master/electrode/utils.py
'''


def populate_map(names):
    d_map = {}
    idx = 0
    for order, names in enumerate(names):
        d_map[order] = []
        for name in names:
            d_map[name] = idx
            d_map[order].append(idx)
            idx += 1
    return d_map


def get_derivative(d, d_map):
    if isinstance(d, int):
        return d_map[d]
    elif isinstance(d, str):
        return d_map[d]
    elif isinstance(d, list):
        return [d_map[name] for name in d]
    else:
        raise TypeError(f"Undefined derivative: {d}")


if __name__ == '__main__':
    # example how-to: this should be implemented in your trap model
    names = [[""]] + [s.split() for s in [
        "x y z",
        "xx xy xz yy yz zz",
    ]]

    d_map = populate_map(names)

    print(names)
    __import__('pprint').pprint(d_map)
    print(get_derivative(1, d_map))
    print(get_derivative('xy', d_map))
    print(get_derivative(['x', 'xy'], d_map))
