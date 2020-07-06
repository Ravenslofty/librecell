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
    """
    Abstract representation of a MOS transistor.
    """

    def __init__(self, channel_type: ChannelType,
                 source_net: str, gate_net: str, drain_net: str,
                 channel_width=None,
                 name: str = 'M?',
                 allow_flip_source_drain: bool = True
                 ):
        """
        params:
        left: Either source or drain net.
        right: Either source or drain net.
        """
        self.name = name
        self.channel_type = channel_type
        self.source_net = source_net
        self.gate_net = gate_net
        self.drain_net = drain_net

        self.channel_width = channel_width

        self.allow_flip_source_drain = allow_flip_source_drain

        # TODO
        self.threshold_voltage = None

    def flipped(self):
        """ Return the same transistor but with left/right terminals flipped.
        """

        assert self.allow_flip_source_drain, "Flipping source and drain is not allowed."

        f = deepcopy(self)
        f.source_net = self.drain_net
        f.drain_net = self.source_net

        return f

    def terminals(self) -> Tuple[Any, Any, Any]:
        """ Return a tuple of all terminal names.
        :return:
        """
        return self.source_net, self.gate_net, self.drain_net

    def __key(self):
        return self.name, self.channel_type, self.source_net, self.gate_net, self.drain_net, self.channel_width, self.threshold_voltage

    def __hash__(self):
        return hash(self.__key())

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __repr__(self):
        return "({}, {}, {})".format(self.source_net, self.gate_net, self.drain_net)


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
