##
## Copyright (c) 2019 Thomas Kramer.
## 
## This file is part of librecell-layout 
## (see https://codeberg.org/tok/librecell/src/branch/master/librecell-layout).
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the CERN Open Hardware License (CERN OHL-S) as it will be published
## by the CERN, either version 2.0 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## CERN Open Hardware License for more details.
## 
## You should have received a copy of the CERN Open Hardware License
## along with this program. If not, see <http://ohwr.org/licenses/>.
## 
## 
##
from enum import Enum

from itertools import islice, tee, chain, product
from copy import deepcopy
from typing import Any, Set, Tuple


class ChannelType(Enum):
    NMOS = 1,
    PMOS = 2


class Transistor:
    def __init__(self, channel_type: ChannelType, left, gate, right, channel_width=None, name='M?'):
        """
        params:
        left: Either source or drain net.
        right: Eigher source or drain net.
        """
        self.name = name
        self.channel_type = channel_type
        self.left = left
        self.gate = gate
        self.right = right

        self.channel_width = channel_width

        # TODO
        self.threshold_voltage = None

    # self.location = None

    def flipped(self):
        """ Return the same transistor but with left/right terminals flipped.
        """

        f = deepcopy(self)
        f.left = self.right
        f.right = self.left

        return f

    def terminals(self) -> Tuple[Any, Any, Any]:
        """ Return a tuple of all terminal names.
        :return:
        """
        return self.left, self.gate, self.right

    def __key(self):
        return self.name, self.channel_type, self.left, self.gate, self.right, self.channel_width, self.threshold_voltage

    def __hash__(self):
        return hash(self.__key())

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __repr__(self):
        return "({}, {}, {})".format(self.left, self.gate, self.right)


class Cell:
    """ Dual row cell.
    """

    def __init__(self, width: int):
        self.width = width
        self.upper = [None] * width
        self.lower = [None] * width

    def get_transistor_locations(self) -> Set[Tuple[Transistor, Tuple[int, int]]]:
        """ Get a list of all transistors together with their location.
        Transistor locations are given on a grid like:

         | (0,1) | (1,1) | ...
         | (0,0) | (1,0) | ...

        Returns
        -------

        Returns a set of (transistor, (x,y)).
        """

        assert len(self.lower) == len(self.upper)

        t = [self.lower, self.upper]
        idx = product(range(self.width), range(2))

        return set((t[y][x], (x, y)) for x, y in idx if t[y][x] is not None)

    def __repr__(self):
        """ Pretty-print
        """

        return (
                " | ".join(['{:^16}'.format(str(t)) for t in self.upper]) +
                "\n" +
                " | ".join(['{:^16}'.format(str(t)) for t in self.lower])
        )
