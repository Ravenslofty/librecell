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


def all_simple_paths_multigraph(G: nx.MultiGraph, source, target, cutoff=None):
    """
    Get edges inclusive keys of all simple paths in a multi graph.
    :param G:
    :param source:
    :param target:
    :param cutoff:
    :return:
    """

    if source not in G:
        raise nx.NodeNotFound('source node %s not in graph' % source)
    if target not in G:
        raise nx.NodeNotFound('target node %s not in graph' % target)
    if source == target:
        return []
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return []
    visited = collections.OrderedDict.fromkeys([source])
    edges = list()  # Store sequence of edges.
    stack = [(((u, v), key) for u, v, key in G.edges(source, keys=True))]
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
                stack.append((((u, v), k) for u, v, k in G.edges(child, keys=True)))
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
    :param vdd_node:
    :param gnd_node:
    :param output_node:
    :param input_names: Ordering of input names.
    :return: sympy.Symbol
    """

    def conductivity_condition(cmos_graph: nx.MultiGraph, source, target):
        all_paths = list(all_simple_paths_multigraph(cmos_graph, source, target))
        transistor_paths = [
            [(gate_net, channel_type) for (net1, net2), (gate_net, channel_type) in path] for path in all_paths
        ]
        f = boolalg.Or(
            *[boolalg.And(
                *(sympy.Symbol(gate) if channel_type == ChannelType.NMOS else ~sympy.Symbol(gate)
                  for gate, channel_type in path
                  )
            ) for path in transistor_paths]
        )
        f = simplify_logic(f)
        return f

    output_at_vdd = conductivity_condition(cmos_graph, output_node, vdd_node)
    output_at_gnd = conductivity_condition(cmos_graph, output_node, gnd_node)

    is_complementary = output_at_gnd.equals(~output_at_vdd)
    print("is complementary: {}".format(is_complementary))
    has_short = satisfiable(output_at_vdd & output_at_gnd)
    print("has short: {}".format(has_short))
    has_tri_state = satisfiable((~output_at_vdd) & (~output_at_gnd))
    print("has tri-state: {}".format(has_tri_state))

    print(~output_at_vdd)

    # pull_up = [
    #     [cmos_graph[a][b] for a, b in pairwise(path)] for path in all_vdd_paths
    # ]
    # print(pull_up)
    # minterms = minterms_from_cmos_graph(cmos_graph, vdd_node, gnd_node, output_node, input_names)
    # dontcares = []
    # input_symbols = [sympy.Symbol(n) for n in input_names]
    # sop = SOPform(input_symbols, minterms, dontcares)
    # sop = sympy.simplify_logic(sop)

    return None


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
    # assert formula.equals(~(a & b)), "Transformation of CMOS graph into formula failed."
