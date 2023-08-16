#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 06/2023
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

from typing import Union, Optional
from pytrans.typing import RoiSize, Bounds, Coords1


class Roi:
    def __init__(self, size: Union[RoiSize, Bounds], center: Optional[Coords1] = None):
        assert len(size) == 3
        self._size = size
        self._bounds = []
        for j in range(len(size)):
            b = size[j]
            if isinstance(b, (tuple, list)) and len(b) == 2:
                pass
            elif isinstance(b, (int, float)):
                if center is None:
                    raise ValueError("center argument is required")
                b = (-float(b) + center[j], float(b) + center[j])
            else:
                raise ValueError(f"Wrong entry in roi.size[{j}]")
            self._bounds.append(b)

    @property
    def bounds(self) -> Bounds:
        return self._bounds
