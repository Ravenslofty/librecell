#
# Copyright (c) 2019-2020 Thomas Kramer.
#
# This file is part of librecell 
# (see https://codeberg.org/tok/librecell).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import networkx as nx
from itertools import product
from typing import Any, Dict, List, Iterable, Tuple
from enum import Enum

import sympy
from sympy.logic import SOPform

from lclayout.data_types import ChannelType


class Signal(Enum):
    LOW = 0
    HIGH = 1
    X = 2
    Z = 3
    SHORT = 4


def extract_output_node_from_cmos_graph(cmos_graph: nx.MultiGraph) -> Any:
    """ Find the output node of the CMOS network
        by finding the only node which connects to both n-channel and p-channel Transistors.
    :param cmos_graph: nx.MultiGraph
        The CMOS network. Edges represent transistors. Each edge key has the form (input name, ChannelType).
    :return: Returns the node of the output signal.
    """

    pn_nodes = [n for n in cmos_graph if
                len(set((ch for _, _, (_, ch) in cmos_graph.edges(n, keys=True)))) > 1
                ]
    assert len(pn_nodes) == 1, "Number of nodes connecting to n-channel and p-channel transistor must be exactly 1."
    output_node = pn_nodes[0]
    return output_node


def evaluate_cmos_graph(cmos_graph: nx.MultiGraph, vdd_node, gnd_node, output_node,
                        input_names: List,
                        inputs: Iterable[List[bool]]) -> List[Signal]:
    """ Simulate a CMOS graph using a simple switch model for the transistors.
    The must be build from a pull-up and a pull-down network.

    :param cmos_graph: Transistor network. Each edge represents a transistor. Edge key must be a tuple (input signal name, ChannelType).
    :param vdd_node: The vdd node in `cmos_graph`.
    :param gnd_node: The gnd node in `cmos_graph`.
    :param output_node: The output node in `cmos_graph`.
    :param input_names: An list of input names.
    :param inputs: Iterable[List[bool]]
        A sequence of input signal assignments. The ordering of the bools must maatch `input_names`.
    :return: Returns a list of outputs for each input assignment. Each element will be Signal.LOW, Signal.HIGH, Signal.Z or Signal.SHORT.
        Signal.Z: Neither pull-up nor pull-down network is conductive.
        Signal.SHORT: Both pull-up and pull-down network are conductive.
    """

    assert nx.is_connected(cmos_graph), '`cmos_graph` must be connected.'
    g = cmos_graph.copy()
    g.remove_node(output_node)
    assert not nx.is_connected(g), '`cmos_graph` is not a pull-up pull-down graph.'

    output_node_exp = extract_output_node_from_cmos_graph(cmos_graph)
    assert output_node == output_node_exp

    def _eval_cmos_graph(cmos_graph, input_mapping):
        g_eval = cmos_graph.copy()

        for a, b, (input_signal, channel_type) in cmos_graph.edges(keys=True):
            inp = input_mapping[input_signal]

            if channel_type == ChannelType.PMOS:
                # Invert for pmos transistors.
                inp = not inp

            if not inp:
                g_eval.remove_edge(a, b)

        is_pulling_up = nx.has_path(g_eval, vdd_node, output_node)
        is_pulling_down = nx.has_path(g_eval, gnd_node, output_node)

        if is_pulling_up and is_pulling_down:
            return Signal.SHORT

        if not is_pulling_up and not is_pulling_down:
            return Signal.Z

        if is_pulling_down:
            return Signal.LOW

        if is_pulling_up:
            return Signal.HIGH

        assert False

    outputs = []
    for inp in inputs:
        input_mapping = {k: v for k, v in zip(input_names, inp)}
        out = _eval_cmos_graph(cmos_graph, input_mapping)
        outputs.append(out)

    return outputs


def test_evaluate_cmos_graph():
    # Create CMOS network of a nand gate and check if `evaluate_cmos_graph` behaves like the nand function.
    g = nx.MultiGraph()
    g.add_edge('vdd', 'output', ('a', ChannelType.PMOS))
    g.add_edge('vdd', 'output', ('b', ChannelType.PMOS))
    g.add_edge('gnd', '1', ('a', ChannelType.NMOS))
    g.add_edge('1', 'output', ('b', ChannelType.NMOS))

    input_names = ['a', 'b']
    n = len(input_names)
    inputs = list(product(*([[0, 1]] * n)))

    # reference logic function
    def f(a, b):
        return not (a and b)

    outputs = evaluate_cmos_graph(g, 'vdd', 'gnd', 'output', input_names, inputs)

    outputs_expected = [Signal.HIGH if f(*inp) else Signal.LOW for inp in inputs]

    assert outputs == outputs_expected


def minterms_from_cmos_graph(cmos_graph: nx.MultiGraph, vdd_node, gnd_node, output_node, input_names: List) \
        -> List[Tuple]:
    """ Simulate a CMOS network with all possible inputs and return a list of inputs that lead to a HIGH output.
    :param cmos_graph:
    :param vdd_node:
    :param gnd_node:
    :param output_node:
    :param input_names:
    :return:
    """
    # Grab input names
    # input_names = [input_name for _,_,(input_name,_) in cmos_graph.edges(keys=True)]

    n = len(input_names)
    inputs = list(product(*([[0, 1]] * n)))

    outputs = evaluate_cmos_graph(cmos_graph, vdd_node, gnd_node, output_node, input_names, inputs)

    return [inp for inp, o in zip(inputs, outputs) if o == Signal.HIGH]


def cmos_graph_to_formula(cmos_graph: nx.MultiGraph, vdd_node, gnd_node, output_node,
                          input_names: List) -> sympy.Symbol:
    """
    Find the boolean formula implemented by the push-pull network `cmos_graph`.
    :param cmos_graph:
    :param vdd_node:
    :param gnd_node:
    :param output_node:
    :param input_names: Ordering of input names.
    :return: sympy.Symbol
    """
    minterms = minterms_from_cmos_graph(cmos_graph, vdd_node, gnd_node, output_node, input_names)
    dontcares = []
    input_symbols = [sympy.Symbol(n) for n in input_names]
    sop = SOPform(input_symbols, minterms, dontcares)
    sop = sympy.simplify_logic(sop)

    return sop


def test_cmos_graph_to_formula():
    # Create CMOS network of a nand gate and check if `evaluate_cmos_graph` behaves like the nand function.
    g = nx.MultiGraph()
    g.add_edge('vdd', 'output', ('a', ChannelType.PMOS))
    g.add_edge('vdd', 'output', ('b', ChannelType.PMOS))
    g.add_edge('gnd', '1', ('a', ChannelType.NMOS))
    g.add_edge('1', 'output', ('b', ChannelType.NMOS))

    input_names = ['a', 'b']

    formula = cmos_graph_to_formula(g, 'vdd', 'gnd', 'output', input_names)

    a, b = sympy.symbols('a b')
    assert formula.equals(~(a & b)), "Transformation of CMOS graph into formula failed."
