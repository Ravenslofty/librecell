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

For each class of cells (latch, single-edge triggered flip-flop, ...) a Extractor class should be created.
The extractor class tries to recognize the cell from an abstract formal description.
For recognizing an unknown cell, all extractor classes are tried until one finds a match.
"""


# def find_clear_and_preset_signals(f: boolalg.Boolean) -> Dict[boolalg.BooleanAtom, Tuple[bool, bool]]:
#     """
#     Find the variables in a boolean formula that can either force the formula to True or False.
#     :param f:
#     :return: Dict[Variable symbol, (is preset, is active high)]
#     """
#     results = dict()
#     atoms = f.atoms(sympy.Symbol)
#     for a in atoms:
#         for v in [False, True]:
#             f.subs({a: v})
#             if f == sympy.true and f != sympy.false:
#                 results[a] = (True, v)
#             elif f == sympy.false and f != sympy.true:
#                 results[a] = (False, v)
#     return results
#
# def test_find_clear_and_preset_signals():
#     a = sympy.Symbol('a')
#     clear = sympy.Symbol('clear')
#     preset = sympy.Symbol('preset')

def find_boolean_isomorphism(a: boolalg.Boolean, b: boolalg.Boolean) -> Optional[
    Dict[boolalg.BooleanAtom, boolalg.BooleanAtom]]:
    """
    Find a one-to-one mapping of the variables in `a` to the variables in `b` such that the both formulas are
    equivalent. Return `None` if there is no such mapping.
    The mapping is not necessarily unique.
    :param a:
    :param b:
    :return:
    """

    a_vars = list(a.atoms(sympy.Symbol))
    b_vars = list(b.atoms(sympy.Symbol))

    if len(a_vars) != len(b_vars):
        return None

    # Do a brute-force search.
    for b_vars_perm in itertools.permutations(b_vars):
        substitution = {old: new for old, new in zip(a_vars, b_vars_perm)}
        a_perm = a.subs(substitution)
        # Check for structural equality first, then for mathematical equality.
        if a_perm == b or bool_equals(a_perm, b):
            return substitution

    return None


def test_find_boolean_isomorphism():
    assert find_boolean_isomorphism(sympy.true, sympy.true) is not None
    assert find_boolean_isomorphism(sympy.true, sympy.false) is None

    a, b, c, x, y, z = sympy.symbols('a b c x y z')
    f = (a & b) | c
    g = (x & y) | z
    mapping = find_boolean_isomorphism(f, g)
    assert mapping == {a: x, b: y, c: z}

    # MUX
    f = (a & c) | (b & ~c)
    g = ~((~x & z) | (~y & ~z))
    mapping = find_boolean_isomorphism(f, g)
    print(mapping)
    assert mapping == {a: x, b: y, c: z}


class Latch:

    def __init__(self):
        self.data_in = None
        self.enable = None  # Write condition / clock.
        self.clear = None  # Clear condition.
        self.preset = None  # Preset condition.

    def __str__(self):
        return self.human_readable_description()

    def human_readable_description(self) -> str:
        return f"""Latch {{
    write data: {self.data_in}
    write enable: {self.enable}
    clear: {self.clear}
    preset: {self.preset}
}}"""


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
        current_nodes = set(output.function.atoms(sympy.Symbol))
        while current_nodes:
            node = current_nodes.pop()
            if node in c.outputs:
                next = c.outputs[node]
                current_nodes.update(next.function.atoms(sympy.Symbol))
            elif node in c.latches:
                latch = c.latches[node]
                assert isinstance(latch, Memory)

                latch_path.append(latch)
                current_nodes = set(latch.data.atoms(sympy.Symbol))
            else:
                # Cannot further resolve the node, must be an input.
                logger.debug(f"Input node: {node}")

        if len(latch_path) != 1:
            logger.debug(f"No latch found in the fan-in tree of the outputs {output.function.atoms()}.")
            return None

        latch = latch_path[0]

        enable_signals = sorted(latch.write_condition.atoms(sympy.Symbol))
        logger.debug(f"Potential clock/set/preset signals {enable_signals}")

        result = Latch()
        result.enable = latch.write_condition
        result.data_in = latch.data

        return result


class SingleEdgeDFF:
    """
    Single-edge triggered delay flip-flop.
    """

    def __init__(self):
        self.clock_signal = None  # Name of the clock signal.
        self.clock_edge_polarity = None  # True = rising, False = falling

        self.data_in = None  # Expression for the input data.
        self.data_out = None  # Name of the non-inverted data output net.
        self.data_out_inv = None  # Name of the inverted data output net (if any).

        self.scan_enable = None  # Name of the scan-enable input.
        self.scan_in = None

        self.async_set_signal = None  # Name of the asynchronous preset signal.
        self.async_set_polarity = None  # Polarity of the signal (False: active low, True: active high).

        self.async_reset_signal = None  # Name of the asynchronous clear signal.
        self.async_reset_polarity = None  # Polarity of the signal (False: active low, True: active high).

    def __str__(self):
        return self.human_readable_description()

    def human_readable_description(self) -> str:

        preset_polarity = ""
        if self.async_set_polarity is not None:
            preset_polarity = "HIGH" if self.async_set_polarity else "LOW"

        clear_polarity = ""
        if self.async_reset_polarity is not None:
            clear_polarity = "HIGH" if self.async_reset_polarity else "LOW"

        return f"""SingleEdgeDFF {{
    clock: {self.clock_signal}
    active clock edge: {"rising" if self.clock_edge_polarity else "falling"}
    output: {self.data_out}
    inverted output: {self.data_out_inv}
    next data: {self.data_in}
    
    asynchronous preset: {self.async_set_signal} {preset_polarity}
    asynchronous clear: {self.async_reset_signal} {clear_polarity}

    scan enable: {self.scan_enable}
    scan input: {self.scan_in}
}}"""


class DFFExtractor:
    def __init__(self):
        pass

    def extract(self, c: AbstractCircuit) -> Optional[SingleEdgeDFF]:
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
        if len(output_nets) not in [1, 2]:
            logger.debug(f"Expect 1 or 2 output nets, found {len(output_nets)}.")
            return None

        # Trace back the output towards the inputs.
        outputs = [c.outputs[n] for n in output_nets]

        # Check that the output is not tri-state.
        for output in outputs:
            if output.is_tristate():
                logger.warning(f"'{output}' is a tri-state output.")
                logger.warning("Can't recognize DFF with tri-state output.")
                return None

        # Trace back the output towards the inputs.
        latch_path = []
        current_nodes = set()
        for output in outputs:
            current_nodes.update(output.function.atoms(sympy.Symbol))
        while current_nodes:
            node = current_nodes.pop()
            if node in c.outputs:
                next = c.outputs[node]
                current_nodes.update(next.function.atoms(sympy.Symbol))
            elif node in c.latches:
                latch = c.latches[node]
                assert isinstance(latch, Memory)

                latch_path.append((latch, node))
                current_nodes = set(latch.data.atoms(sympy.Symbol))
            else:
                # Cannot further resolve the node, must be an input.
                logger.debug(f"Input node: {node}")

        # Find trigger condition of the flip-flop.
        latch1, latch1_output_node = latch_path[1]
        latch2, latch2_output_node = latch_path[0]

        # Find condition such that the FF is in normal operation mode.
        # I.e. no set or reset is active.
        ff_normal_condition = simplify_logic(latch1.write_condition ^ latch2.write_condition)
        logger.debug(f"FF normal-mode condition: {ff_normal_condition}")

        # Find clock inputs of the latches.
        latch1_write_normal = simplify_with_assumption(assumption=ff_normal_condition, formula=latch1.write_condition)
        latch2_write_normal = simplify_with_assumption(assumption=ff_normal_condition, formula=latch2.write_condition)

        logger.debug(f"Latch clocks: {latch1_write_normal}, {latch2_write_normal}")

        clock_signals = set(latch1_write_normal.atoms(sympy.Symbol)) | set(latch2_write_normal.atoms(sympy.Symbol))
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
        dff = SingleEdgeDFF()
        dff.clock_signal = clock_signal
        dff.clock_edge_polarity = active_edge_polarity

        # == Detect asynchronous set/reset signals ==
        potential_set_reset_signals = list((set(latch1.write_condition.atoms(sympy.Symbol))
                                            | set(latch2.write_condition.atoms(sympy.Symbol))) - {clock_signal}
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
                assert bool_equals(wc2, True)
                if data2:
                    logger.info(f"{signal} is an async SET/PRESET signal, active {'high' if signal_value else 'low'}.")
                    async_set_signals.append((signal, signal_value))
                else:
                    logger.info(f"{signal} is an async RESET/CLEAR signal, active {'high' if signal_value else 'low'}.")
                    async_reset_signals.append((signal, signal_value))

            # TODO: Find out what happens when all asynchronous set/reset signals are active at the same time.
            all_sr_active_wc = latch2.write_condition.subs(sr_enabled_values)
            all_sr_active_data = latch2.data.subs(sr_enabled_values)
            # TODO

            if len(async_set_signals) > 1:
                logger.error(f"Multiple async SET/PRESET signals: {async_set_signals}")
            if len(async_reset_signals) > 1:
                logger.error(f"Multiple async RESET/CLEAR signals: {async_reset_signals}")

            # Store the results in the flip-flop description object.
            if async_set_signals:
                dff.async_set_signal, dff.async_set_polarity = async_set_signals[0]
            if async_reset_signals:
                dff.async_reset_signal, dff.async_reset_polarity = async_reset_signals[0]

        ff_output_data = []
        for output in outputs:
            # Find data that gets written into the flip-flop in normal operation mode.
            ff_data_next = simplify_with_assumption(ff_normal_condition, output.function)
            # Eliminate Set/Reset.
            latch1_data_normal = simplify_with_assumption(ff_normal_condition, latch1.data)
            latch2_data_normal = simplify_with_assumption(ff_normal_condition, latch2.data)
            # Eliminate clock.
            latch1_data_normal = latch1_data_normal.subs({clock_signal: not active_edge_polarity})
            latch2_data_normal = latch2_data_normal.subs({clock_signal: active_edge_polarity})

            # Resolve through second latch.
            ff_data_next = ff_data_next.subs({
                latch2_output_node: latch2_data_normal,
                clock_signal: active_edge_polarity
            })

            # Resolve through first latch.
            ff_data_next = ff_data_next.subs({
                latch1_output_node: latch1_data_normal,
                clock_signal: not active_edge_polarity
            })

            logger.debug(f"Flip-flop output data: {ff_data_next}")
            ff_output_data.append(ff_data_next)

        if len(ff_output_data) == 1:
            # FF has only one output.
            logger.info(f"{output_nets[0]} = {ff_output_data[0]}")
            out_data = ff_output_data[0]
            dff.data_in = out_data
            dff.data_out = output_nets[0]
        elif len(ff_output_data) == 2:
            # Check if the outputs are inverted versions.

            out1, out2 = output_nets
            out1_data, out2_data = ff_output_data

            # Check if the outputs are inverses.
            if out1_data == boolalg.Not(out2_data):
                logger.debug("Outputs are inverses of each other.")
            else:
                logger.warning("If a flip-flop has two outputs, then need to be inverses of each other.")
                return None

            # Find the inverted and non-inverted output.
            if type(out1_data) == boolalg.Not:
                assert type(out2_data) != boolalg.Not
                out_inv_net = out1
                out_net = out2
                out_inv_data = out1_data
                out_data = out2_data
            else:
                assert type(out2_data) == boolalg.Not
                out_inv_net = out2
                out_net = out1
                out_inv_data = out2_data
                out_data = out1_data

            logger.info(f"Non-inverted output: {out_net} = {out_data}")
            logger.info(f"Inverted output: {out_inv_net} = {out_inv_data}")

            dff.data_in = out_data
            dff.data_out = out_net
            dff.data_out_inv = out_inv_net
        else:
            assert False

        # Analyze boolean formula of the next flip-flop state.
        # In the simplest case of a D-flip-flop this is just one variable.
        # But there might also be a scan-chain multiplexer or synchronous
        # clear/preset logic.
        data_variables = out_data.atoms(sympy.Symbol)
        num_variables = len(data_variables)

        if num_variables == 1:
            # Simple. Only a data variable, no clear/preset/scan.
            pass
        elif num_variables == 2:
            # There might be a synchronous clear/preset.
            logger.debug(f"Try to detect a synchronous clear or preset from the two data signals {data_variables}.")
            # It is not possible to distinguish data input from clear/preset based only on
            # the boolean formula.
            d, clear, preset = sympy.symbols('d clear preset')
            with_clear = d & clear
            with_preset = d | preset
            if find_boolean_isomorphism(out_data, with_clear) is not None:
                logger.info("Detected synchronous clear.")
            if find_boolean_isomorphism(out_data, with_preset) is not None:
                logger.info("Detected synchronous preset.")
        elif num_variables == 3:
            # Either synchronous clear/preset or scan.
            logger.debug("Try to detect scan-mux.")
            d, scan_enable, scan_in = sympy.symbols('d scan_enable scan_in')

            scan_mux = (scan_in & scan_enable) | (d & ~scan_enable)

            mapping = find_boolean_isomorphism(scan_mux, out_data)
            if mapping is not None:
                logger.info(f"Detected scan-chain mux: {mapping}")
                dff.scan_in = mapping[scan_in]
                dff.scan_enable = mapping[scan_enable]

        else:
            logger.warning(f"Flip-flop data depends on {num_variables} variables."
                           f" Cannot distinguish scan-mux, clear, preset.")

        return dff


def extract_sequential_circuit(c: AbstractCircuit) -> Optional[Union[Latch, SingleEdgeDFF]]:
    logger.debug("Recognize sequential cells.")
    logger.debug(f"Combinational formulas: {c.outputs}")
    logger.debug(f"Latches: {c.latches}")

    extractors = [LatchExtractor(), DFFExtractor()]

    for ex in extractors:
        result = ex.extract(c)
        if result is not None:
            return result
    return None
