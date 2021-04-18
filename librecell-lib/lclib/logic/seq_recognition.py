##
## Copyright (c) 2021 Thomas Kramer.
##
## This file is part of librecell-lib
## (see https://codeberg.org/tok/librecell/src/branch/master/librecell-lib).
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.
##

from .functional_abstraction import *

import itertools
import networkx as nx
from typing import Any, Dict, List, Iterable, Tuple, Set, Optional, Union
from enum import Enum
import collections
import sympy
from sympy.logic import satisfiable, simplify_logic as sympy_simplify_logic
from sympy.logic import boolalg

from lclayout.data_types import ChannelType
import logging

# logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

"""
Recognize sequential cells based on the output of the `functional_abstraction.analyze_circuit_graph()` function.
"""


class Latch:

    def __init__(self):
        self.data_in = None
        self.enable = None  # Write condition.


class LatchExtractor:
    def __init__(self):
        pass

    def extract(self, c: AbstractCircuit) -> Optional[Latch]:
        """
        Try to recognize a latch based on the abstract circuit representation.
        :param c:
        :return:
        """
        logger.debug("Try to extract a latch.")
        if len(c.latches) != 1:
            logger.debug(f"Not a latch. Wrong number of latches. Need 1, found {len(c.latches)}.")
            return None

        return None


class DFF:
    pass


class DFFExtractor:
    def __init__(self):
        pass

    def extract(self, c: AbstractCircuit) -> Optional[Latch]:
        """
        Try to recognize a D-flip-flop based on the abstract circuit representation.
        :param c:
        :return:
        """
        logger.debug("Try to extract a D-flip-flop.")
        if len(c.latches) != 2:
            logger.debug(f"Not a flip-flop. Wrong number of latches. Need 2, found {len(c.latches)}.")
            return None

        output_nets = list(c.outputs.keys())
        if len(output_nets) != 1:
            logger.debug(f"Expect 1 output net, found {len(output_nets)}.")
            return None

        # Trace back the output towards the inputs.
        output_net = output_nets[0]
        output = c.outputs[output_net]

        # Check that the output is not tri-state.
        if output.is_tristate():
            logger.debug("Can't recognize DFF with tri-state output.")
            return None

        # Trace back the output towards the inputs.
        latch_path = []
        current_nodes = set(output.function.atoms())
        while current_nodes:
            node = current_nodes.pop()
            if node in c.outputs:
                next = c.outputs[node]
                current_nodes.update(next.function.atoms())
            elif node in c.latches:
                latch = c.latches[node]
                assert isinstance(latch, Memory)

                latch_path.append(latch)
                current_nodes = set(latch.data.atoms())
            else:
                # Cannot further resolve the node, must be an input.
                logger.debug(f"Input node: {node}")

        # Find trigger condition of the flip-flop.
        latch1 = latch_path[1]
        latch2 = latch_path[0]

        # Find condition such that the FF is in normal operation mode.
        # I.e. no set or reset is active.
        ff_normal_condition = simplify_logic(latch1.write_condition ^ latch2.write_condition)
        logger.debug(f"FF normal-mode condition: {ff_normal_condition}")

        # Find clock inputs of the latches.
        latch1_write = simplify_with_assumption(assumption=ff_normal_condition, formula=latch1.write_condition)
        latch2_write = simplify_with_assumption(assumption=ff_normal_condition, formula=latch2.write_condition)

        logger.debug(f"Latch clocks: {latch1_write}, {latch2_write}")

        clock_signals = set(latch1_write.atoms()) | set(latch2_write.atoms())
        if len(clock_signals) != 1:
            logger.warning(f"Clock must be a single signal. Found: {clock_signals}")
            return None

        clock_signal = clock_signals.pop()
        logger.info(f"Clock: {clock_signal}")

        assert not satisfiable(boolalg.Equivalent(latch1_write, latch2_write)), "Clock polarities of latches must be complementary."

        # Find polarity of the active clock-edge.
        edge_polarity = latch2_write.subs({clock_signal: True})
        logger.info(f"Edge polarity: {'rising' if edge_polarity else 'falling'}")

        return None


def extract_sequential_circuit(c: AbstractCircuit) -> Optional[Union[Latch, DFF]]:
    logger.debug("Recognize sequential cells.")
    logger.debug(f"Combinational formulas: {c.outputs}")
    logger.debug(f"Latches: {c.latches}")

    extractors = [LatchExtractor(), DFFExtractor()]

    for ex in extractors:
        result = ex.extract(c)
        if result is not None:
            return result
    return None
