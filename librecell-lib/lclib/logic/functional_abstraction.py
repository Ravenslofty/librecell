##
## Copyright (c) 2019 Thomas Kramer.
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

import networkx as nx
from networkx.utils import pairwise
from itertools import product
from typing import Any, Dict, List, Iterable, Tuple
from enum import Enum
import collections
import sympy
from sympy.logic import simplify_logic, satisfiable
from sympy.logic import boolalg
from sympy.logic import SOPform

from lclayout.data_types import ChannelType


def all_simple_paths_multigraph(graph: nx.MultiGraph, source, target, cutoff=None):
    """
    Enumerate all simple paths (no node occurs more than once) from source to target.
    Yields edges inclusive keys of all simple paths in a multi graph.
    :param graph: 
    :param source:
    :param target:
    :param cutoff:
    :return:
    """

    if source not in graph:
        raise nx.NodeNotFound('source node %s not in graph' % source)
    if target not in graph:
        raise nx.NodeNotFound('target node %s not in graph' % target)
    if source == target:
        return []
    if cutoff is None:
        cutoff = len(graph) - 1
    if cutoff < 1:
        return []
    visited = collections.OrderedDict.fromkeys([source])
    edges = list()  # Store sequence of edges.
    stack = [(((u, v), key) for u, v, key in graph.edges(source, keys=True))]
    while stack:
        children = stack[-1]
        (child_source, child), child_key = next(children, ((None, None), None))
        if child is None:
            stack.pop()
            visited.popitem()
            edges = edges[:-1]
        elif len(visited) < cutoff:
            if child == target:
                yield list(edges) + [((child_source, child), child_key)]
            elif child not in visited:
                visited[child] = None
                edges.append(((child_source, child), child_key))
                stack.append((((u, v), k) for u, v, k in graph.edges(child, keys=True)))
        else:  # len(visited) == cutoff:
            count = (list(children) + [child]).count(target)
            for i in range(count):
                yield list(edges) + [((child_source, child), child_key)]
            stack.pop()
            visited.popitem()
            edges = edges[:-1]


def cmos_graph_to_formula(cmos_graph: nx.MultiGraph, vdd_node, gnd_node, output_node,
                          input_names: List) -> sympy.Symbol:
    """
    Find the boolean formula implemented by the push-pull network `cmos_graph`.
    :param cmos_graph:
    :param vdd_node: Name of VDD supply node.
    :param gnd_node: Name of GND supply node.
    :param output_node:
    :param input_names: Ordering of input names.
    :return: sympy.Symbol
    """

    def conductivity_condition(cmos_graph: nx.MultiGraph, source, target):
        """
        Find a boolean equation that evaluates to true iff there is a conductive path from `source` to `target`
        given the input signals.
        :param cmos_graph:
        :param source:
        :param target:
        :return: Boolean function. (sympy.Symbol)
        """
        # Get all simple paths from source to target.
        all_paths = list(all_simple_paths_multigraph(cmos_graph, source, target))
        transistor_paths = [
            [(gate_net, channel_type) for (net1, net2), (gate_net, channel_type) in path] for path in all_paths
        ]
        # There is a conductive path if at least one of the paths is conductive -> Or
        # A path is conductive if all edges (transistors) along the path are conductive -> And
        f = boolalg.Or(
            *[boolalg.And(
                *(sympy.Symbol(gate) if channel_type == ChannelType.NMOS else ~sympy.Symbol(gate)
                  for gate, channel_type in path
                  )
            ) for path in transistor_paths]
        )
        # Try to simplfy the boolean expression.
        f = simplify_logic(f)
        return f

    def bool_equals(f1, f2):
        """
        Check equality of two boolean formulas.
        :param f1:
        :param f2:
        :return:
        """
        return not satisfiable(f1 ^ f2)

    output_at_vdd = conductivity_condition(cmos_graph, output_node, vdd_node)
    output_at_gnd = conductivity_condition(cmos_graph, output_node, gnd_node)
    print(output_at_vdd)
    print(output_at_gnd)
    print(~output_at_vdd)
    is_complementary = bool_equals(output_at_gnd, ~output_at_vdd)
    print("is complementary: {}".format(is_complementary))
    short_condition = output_at_vdd & output_at_gnd
    has_short = satisfiable(short_condition)
    print("has short: {}".format(has_short))
    # print("has when: {}".format(short_condition))
    tri_state_condition = simplify_logic((~output_at_vdd) & (~output_at_gnd))
    has_tri_state = satisfiable(tri_state_condition)
    print("has tri-state: {}".format(has_tri_state))
    print("tri-state when: {}".format(tri_state_condition))

    f_out = simplify_logic(output_at_vdd)
    print(output_node, "=", f_out)

    return f_out


def test_cmos_graph_to_formula():
    # Create CMOS network of a nand gate and check if `evaluate_cmos_graph` behaves like the nand function.
    g = nx.MultiGraph()
    g.add_edge('vdd', 'output', ('a', ChannelType.PMOS))
    g.add_edge('vdd', 'output', ('b', ChannelType.PMOS))
    g.add_edge('gnd', '1', ('a', ChannelType.NMOS))
    g.add_edge('1', 'output', ('b', ChannelType.NMOS))

    input_names = ['a', 'b']

    formula = cmos_graph_to_formula(g, 'vdd', 'gnd', 'output', input_names)

    # Verify that the deduced formula equals a NAND.
    a, b = sympy.symbols('a b')
    nand = ~(a & b)
    assert formula.equals(nand), "Transformation of CMOS graph into formula failed."
