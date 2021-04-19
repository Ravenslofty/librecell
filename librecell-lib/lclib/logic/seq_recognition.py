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

    def __init__(self):
        self.clock_signal = None
        self.clock_edge_polarity = None

        self.data_in = None
        self.data_out = None

        self.scan_enable = None
        self.scan_in = None

        self.async_set_signal = None
        self.async_set_active = None

        self.async_reset_signal = None
        self.async_reset_active = None


class DFFExtractor:
    def __init__(self):
        pass

    def extract(self, c: AbstractCircuit) -> Optional[Latch]:
        """
        Try to recognize a single-edge triggered D-flip-flop based on the abstract circuit representation.
        :param c:
        :return:
        """
        logger.debug("Try to extract a D-flip-flop.")
        if len(c.latches) != 2:
            logger.debug(f"Not a flip-flop. Wrong number of latches. Need 2, found {len(c.latches)}.")
            return None

        output_nets = list(c.output_pins)
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
        latch1_write_normal = simplify_with_assumption(assumption=ff_normal_condition, formula=latch1.write_condition)
        latch2_write_normal = simplify_with_assumption(assumption=ff_normal_condition, formula=latch2.write_condition)

        logger.debug(f"Latch clocks: {latch1_write_normal}, {latch2_write_normal}")

        clock_signals = set(latch1_write_normal.atoms()) | set(latch2_write_normal.atoms())
        if len(clock_signals) != 1:
            logger.warning(f"Clock must be a single signal. Found: {clock_signals}")
            return None

        clock_signal = clock_signals.pop()
        logger.info(f"Clock signal: {clock_signal}")

        assert not satisfiable(boolalg.Equivalent(latch1_write_normal, latch2_write_normal)), \
            "Clock polarities of latches must be complementary."

        # Find polarity of the active clock-edge.
        active_edge_polarity = latch2_write_normal.subs({clock_signal: True})

        # Sanity check: The both latches must be transparent for the opposite phases of the clock.
        assert active_edge_polarity == latch1_write_normal.subs({clock_signal: False}), \
            "Latches must be transparent in opposite phases of the clock."

        logger.info(f"Active edge polarity: {'rising' if active_edge_polarity else 'falling'}")

        # Assemble D-flip-flop description object.
        dff = DFF()
        dff.clock_signal = clock_signal
        dff.clock_edge_polarity = active_edge_polarity

        # == Detect asynchronous set/reset signals ==
        potential_set_reset_signals = list((set(latch1.write_condition.atoms())
                                            | set(latch2.write_condition.atoms())) - {clock_signal}
                                           )
        logger.debug(
            f"Potential asynchronous set/reset signals: {sorted(potential_set_reset_signals, key=lambda n: n.name)}")

        if potential_set_reset_signals:
            # More than 2 asynchronous set/reset signals are not supported.
            if len(potential_set_reset_signals) > 2:
                logger.error(f"Cannot recognize flip-flops with more than 2 asynchronous set/reset signals "
                             f"(found {potential_set_reset_signals}).")
                return None

            # Find signal values such that the FF is in normal operation mode.
            inactive_set_reset_models = list(satisfiable(ff_normal_condition, all_models=True))
            if len(inactive_set_reset_models) != 1:
                logger.warning(f"There's not exactly one signal assignment such that the FF is in normal mode: "
                               f"{inactive_set_reset_models}")
            assert inactive_set_reset_models, f"Normal operation condition is not satisfiable: {ff_normal_condition}."

            # Determine polarity of the set/reset signals.
            sr_disabled_values = inactive_set_reset_models[0]
            logger.info(f"Set/reset disabled when: {sr_disabled_values}")
            sr_enabled_values = {k: not v for k, v in sr_disabled_values.items()}
            logger.info(f"Set/reset enabled when: {sr_enabled_values}")

            async_set_signals = []
            async_reset_signals = []

            # Find if the signals are SET or RESET signals.
            # Set just one to active and observe the flip-flop output.
            for signal in potential_set_reset_signals:
                one_active = sr_disabled_values.copy()
                signal_value = sr_enabled_values[signal]
                one_active.update({signal: signal_value})

                wc2 = latch2.write_condition.subs(one_active)
                data2 = latch2.data.subs(one_active)
                assert wc2 == True
                if data2:
                    logger.info(f"{signal} is a SET/PRESET signal, active {'high' if signal_value else 'low'}.")
                    async_set_signals.append((signal, signal_value))
                else:
                    logger.info(f"{signal} is a RESET/CLEAR signal, active {'high' if signal_value else 'low'}.")
                    async_reset_signals.append((signal, signal_value))

            # Find out what happens when all asynchronous set/reset signals are active at the same time.
            all_sr_active_wc = latch2.write_condition.subs(sr_enabled_values)
            all_sr_active_data = latch2.data.subs(sr_enabled_values)
            # TODO

            if len(async_set_signals) > 1:
                logger.error(f"Multiple async SET/PRESET signals: {async_set_signals}")
            if len(async_reset_signals) > 1:
                logger.error(f"Multiple async RESET/CLEAR signals: {async_reset_signals}")

            # Store the results in the flip-flop description object.
            if async_set_signals:
                dff.async_set_signal, dff.async_set_active = async_set_signals[0]
            if async_reset_signals:
                dff.async_reset_signal, dff.async_reset_active = async_reset_signals[0]

        return dff


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
