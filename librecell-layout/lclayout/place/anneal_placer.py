#
# Copyright 2021 Dan Ravensloft.
#
# This source describes Open Hardware and is licensed under the CERN-OHL-S v2.
#
# You may redistribute and modify this documentation and make products using it
# under the terms of the CERN-OHL-S v2 (https:/cern.ch/cern-ohl).
# This documentation is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY,
# INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
# Please see the CERN-OHL-S v2 for applicable conditions.
#
# Source location: https://codeberg.org/tok/librecell
#
from .place import TransistorPlacer

from ..data_types import *

from random import Random
from typing import Iterable, List, Tuple

import logging

logger = logging.getLogger(__name__)

def _assemble_cell(lower_row: List[Transistor], upper_row: List[Transistor]) -> Cell:
    """ Build a Cell object from a nmos and pmos row.
    :param lower_row:
    :param upper_row:
    :return:
    """
    width = max(len(lower_row), len(upper_row))
    cell = Cell(width)
    for i, t in enumerate(upper_row):
        cell.upper[i] = t

    for i, t in enumerate(lower_row):
        cell.lower[i] = t
    return cell

def _legalise(lower_row: List[Transistor], upper_row: List[Transistor]) -> Tuple[List[Transistor], List[Transistor]]:
    """Legalise transistor rows.
    :param lower_row:
    :param upper_row:
    :return:
    """
    while len(lower_row) < len(upper_row):
        lower_row.append(None)
    while len(upper_row) < len(lower_row):
        upper_row.append(None)

    if len(lower_row) == 0:
        return upper_row, lower_row

    # Constrain transistors to the correct row.
    if lower_row[0] is not None and lower_row[0].channel_type == ChannelType.PMOS:
        if upper_row[0] is not None and upper_row[0].channel_type != ChannelType.PMOS:
            return _legalise(upper_row, lower_row)
        else:
            return _legalise([None] + lower_row[1:], upper_row[0:1] + lower_row[0:1] + upper_row[1:])

    if upper_row[0] is not None and upper_row[0].channel_type == ChannelType.NMOS:
        if lower_row[0] is not None and lower_row[0].channel_type != ChannelType.NMOS:
            return _legalise(upper_row, lower_row)
        else:
            return _legalise(lower_row[0:1] + upper_row[0:1] + lower_row[1:], [None] + upper_row[1:])

    # Insert diffusion gaps for non-matching nets.
    if len(lower_row) >= 2 and lower_row[0] is not None and lower_row[1] is not None and lower_row[0].drain_net != lower_row[1].source_net:
        return _legalise(lower_row[0:1] + [None] + lower_row[1:], upper_row)

    if len(upper_row) >= 2 and upper_row[0] is not None and upper_row[1] is not None and upper_row[0].drain_net != upper_row[1].source_net:
        return _legalise(lower_row, upper_row[0:1] + [None] + upper_row[1:])

    lower_tail, upper_tail = _legalise(lower_row[1:], upper_row[1:])
    return (lower_row[0:1] + lower_tail, upper_row[0:1] + upper_tail)


class RandomPlacer(TransistorPlacer):
    def __init__(self):
        # For reproducibility, seed the generator with a fixed constant.
        self.rand = Random(1)

    def place(self, transistors: Iterable[Transistor]) -> Cell:
        """Place transistors randomly.
        :param transistors:
        :return:
        """
        # First, shuffle the order of the transistors.
        self.rand.shuffle(transistors)

        # For additional randomness, shuffle transistor sources and drain.
        for transistor in transistors:
            if self.rand.choice([False, True]):
                transistor.source_net, transistor.drain_net = transistor.drain_net, transistor.source_net

        # Legalise placement by inserting diffusion gaps.
        l = len(transistors)
        lower_row, upper_row = _legalise(transistors[:l//2], transistors[l//2:])

        return _assemble_cell(lower_row, upper_row)


class HillClimbPlacer(TransistorPlacer):
    def place(self, transistors: Iterable[Transistor]) -> Cell:
        placer = RandomPlacer()
        start = placer.place(transistors)

        pass


class AnnealPlacer(TransistorPlacer):
    def place(self, transistors: Iterable[Transistor]) -> Cell:
        nmos = [t for t in transistors if t.channel_type == ChannelType.NMOS]
        pmos = [t for t in transistors if t.channel_type == ChannelType.PMOS]

        pass
