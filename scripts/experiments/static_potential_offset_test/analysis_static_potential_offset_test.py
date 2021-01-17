#!/usr/bin/env python3
# Analysis of data acquired during static_potential_offset_test experiment

import sys
sys.path.append("../../")
from pytrans import *

def analyse():
    pass

if __name__ == "__main__":
    expected_freqs = np.array([
        1598, 1589, 1606, 1589, 1607, 1589, 1607, 1589, 1607, 1589, 1607, 1589, 1607, 1592, 1604, 1592, 1604, 1586, 1610, 1586, 1610, 1589, 1607, 1589.5, 1606.5, 1590, 1606, 1592, 1604, 1592, 1604
    ])
    
    measured_freqs = np.array([
        1642, 1627, 1656, 1634, 1649, 1630, 1652, 1632, 1650, 1635, 1649, 1642, 1642, 1638, 1646, 1637, 1647, 1630, 1653, 1627, 1656, 1633, 1651, 1634, 1651, 1634, 1651, 1635, 1650, 1638, 1648, 1637, 1648
    ])

    
