#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 08/2021
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

""" Various tools to easily index arrays
Inspired by:
- https://github.com/nist-ionstorage/electrode/blob/master/electrode/utils.py
- https://stackoverflow.com/a/43090200
"""


import re
import numpy as np

re_ix = r"(\[[\s\d,]+\]|[-\d:]+)"


def gradient_matrix(n):
    M = np.zeros((n, n))
    M += np.diagflat([1] * (n - 1), k=1) / 2
    M += np.diagflat([-1] * (n - 1), k=-1) / 2
    M[0, [0, 1]] = -1, 1
    M[-1, [-2, -1]] = -1, 1
    return M

def diff_matrix(n):
    M = np.zeros((n-1,n))
    M+= np.concatenate([np.diag([-1.] * (n-1)),np.zeros((n-1,1)).reshape([-1,1])],axis=1)
    M+= np.concatenate([np.zeros((n-1,1)).reshape([-1,1]),np.diag([+1.] * (n-1))],axis=1)
    return M

def populate_map(names):
    d_map = {}
    for order, names in enumerate(names):
        idx = 0
        d_map[order] = []
        for name in names:
            d_map[name] = idx
            d_map[order].append(idx)
            idx += 1
    return d_map


d_names = [[""]] + [s.split() for s in [
    "x y z",
    "xx xy xz yx yy yz zx zy zz",
]]

d_map = populate_map(d_names)


def get_derivative(d, d_map=d_map):
    if isinstance(d, int):
        return d_map[d]
    elif isinstance(d, str):
        return d_map[d]
    elif isinstance(d, list):
        return [d_map[name] for name in d]
    else:
        raise TypeError(f"Undefined derivative: {d}")


def parse_indexing_string(indexing: str):
    """Parse a numpy-style indexing string to a tuple of slices
    freely inspired from https://stackoverflow.com/a/43090200

    Args:
      indexing: str: can contain digits, commas, square brackets, colons, and whitespace

    Returns:
      slices: tuple of slice, int, or list of int

    Examples:
      '[0, 1], 10, ::-1' -> ([0, 1], 10, slice(None, None, -1))
      ':, :, 1' -> (slice(None, None, None), slice(None, None, None), 1)
      '0:-4:10,::-1,112,[0,10,18]' -> (slice(0, -4, 10), slice(None, None, -1), 112, [0, 10, 18])
      ':, 3,...,10' -> invalid string
      ':, a, 11' -> invalid string
      '[:,1], ::' -> invalid string
      '1, 3, [, 11' -> invalid string
    """
    rem = re.sub(re_ix, '', indexing)
    if not set(rem) <= {',', ' '}:
        # there should be no other characters left after matching
        raise ValueError("invalid string")
    slices = []
    for match in re.findall(re_ix, indexing):
        if '[' in match:
            sl = list(map(int, match.strip('[]').split(',')))
        elif ':' in match:
            sl = slice(*(int(i) if i else None for i in match.strip().split(':')))
        else:
            sl = int(match)
        slices.append(sl)
    return tuple(slices)


def parse_indexing(indexing):
    if isinstance(indexing, (slice, tuple, list, int)):
        return indexing
    elif isinstance(indexing, str):
        return parse_indexing_string(indexing)
    else:
        raise TypeError("Incorrect type of indexing")


if __name__ == '__main__':
    strings = """[0, 1], 10, ::-1
:, :, 1
0:-4:10,::-1,112,[0,10,18]
:, 3,...,10
:, a, 11
[:,1], ::
1, 3, [, 11""".split('\n')

    for string in strings:
        print(f"'{string}'", end=' -> ')
        try:
            print(parse_indexing_string(string))
        except ValueError as e:
            print(e)

    print(d_names)
    __import__('pprint').pprint(d_map)
    print(get_derivative(1, d_map))
    print(get_derivative('xy', d_map))
    print(get_derivative(['x', 'xy'], d_map))
