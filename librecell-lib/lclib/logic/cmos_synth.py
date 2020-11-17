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
import sympy
from sympy.logic.boolalg import to_cnf, to_dnf
from sympy.logic import POSform, SOPform
from sympy.utilities.lambdify import lambdify

from itertools import chain, product, count, combinations

import networkx as nx

from typing import Any, Tuple, List

from lclayout.data_types import Transistor, ChannelType

import logging

logger = logging.getLogger(__name__)


def test_convert_to_dnf():
    """
    Example for finding the DNF of a boolean formula.
    :return:
    """
    a, b, s = sympy.symbols('a b s')

    z = (a & s) | (b & ~s)

    dnf1 = to_dnf(z, simplify=True)
    dnf2 = sympy.simplify_logic(z, form='dnf')

    assert dnf1 == dnf2

    x = z

    def print_tree(expr: sympy.Symbol, depth=0):
        tab = ' ' * depth
        print(tab, expr.func, expr.args)
        for arg in expr.args:
            print_tree(arg, depth + 1)

    # print_tree(x)


def _formula_to_cmos_network(f: sympy.Symbol):
    counter = count(1)

    def _formula_to_pull_network(f: sympy.Symbol, channel_type: ChannelType) -> nx.MultiGraph:
        """ Create the pull-up or pull-down network of a boolean forumla `f` in DNF or CNF.

        :param f: Boolean formula in DNF or CNF.
        :param channel_type:
        :return: Returns a nx.MultiGraph representing the CMOS network. Each edge represents a transistor.
                Each node represents a net. Special nets are 'gnd', 'vdd' and 'z'. 'z' is the output net of the network.
        """

        if isinstance(f, sympy.And) or isinstance(f, sympy.Or):

            graphs = [_formula_to_pull_network(a, channel_type) for a in f.args]

            parallel = isinstance(f, sympy.Or)
            serial = isinstance(f, sympy.And)

            assert serial ^ parallel

            if not parallel:
                # Put sub graphs in series
                a = graphs[0]
                for b in graphs[1:]:
                    joint = next(counter)
                    mapping_a = {'power': joint}
                    mapping_b = {'output': joint}
                    a = nx.relabel_nodes(a, mapping_a)
                    b = nx.relabel_nodes(b, mapping_b)
                    a = nx.compose(a, b)

                graph = a
            elif parallel:
                # Put sub graphs in parallel
                graph = nx.compose_all(graphs)
        else:

            assert isinstance(f, sympy.Not) or isinstance(f, sympy.Symbol), TypeError("Expected And, Or or Symbol.")

            if channel_type == ChannelType.PMOS:
                f = sympy.simplify(sympy.Not(f))

            graph = nx.MultiGraph()
            graph.add_edge('power', 'output', key=(f, channel_type))

        assert graph is not None
        return graph

    normal_form = 'dnf'
    nf = sympy.simplify_logic(f, form=normal_form)
    nf_neg = sympy.simplify_logic(~f, form=normal_form)

    nmos_graph = _formula_to_pull_network(nf_neg, ChannelType.NMOS)
    pmos_graph = _formula_to_pull_network(nf, ChannelType.PMOS)

    nmos_graph = nx.relabel_nodes(nmos_graph, {'power': 'gnd'})
    pmos_graph = nx.relabel_nodes(pmos_graph, {'power': 'vdd'})

    cmos_graph = nx.compose(nmos_graph, pmos_graph)

    return cmos_graph


def _cmos_graph_to_transistors(input_signals, cmos_graph: nx.MultiGraph) -> List[Transistor]:
    """
    Convert a labeled multigraph into the corresponding transistor network.
    :param input_signals:
    :param cmos_graph:
    :return:
    """
    all_nets = set()
    # Create normalized net names
    for source, drain, (gate, channel_type) in cmos_graph.edges(keys=True):
        all_nets.add(source)
        all_nets.add(drain)
        all_nets.add(gate)

    cnt = count(1)
    net_name_mapping = dict()
    for net in all_nets:
        if isinstance(net, str):
            name = net
        elif net in input_signals:
            name = str(net)
        elif net == 'output':
            name = 'output'
        else:
            name = '%d' % (next(cnt))

        net_name_mapping[net] = name

    cmos_graph = nx.relabel_nodes(cmos_graph, net_name_mapping, copy=True)

    # Create transistors
    transistors = []
    for source, drain, (gate, channel_type) in cmos_graph.edges(keys=True):
        t = Transistor(channel_type, source, gate, drain, channel_width=1)
        transistors.append(t)

    return transistors


def synthesize_circuit(f: sympy.Symbol) -> nx.MultiGraph:
    """ Synthesize a CMOS network implementing the boolean formula `f`.
    :param f:
    :return: A nx.MultiGraph representing the transistor network.
    """
    cmos_graph = _formula_to_cmos_network(f)

    # Create cmos networks for all gate inputs which are not just a signal.
    # For example this generates inverted input signals when needed.
    input_signals = set(f.atoms())
    input_sources = set()

    for _, _, (input_formula, channel_type) in cmos_graph.edges(keys=True):
        if input_formula not in input_signals and input_formula not in input_sources:
            input_circuit = _formula_to_cmos_network(input_formula)
            input_sources.add(input_circuit)

    # Merge input networks into full graph.
    cmos_graph = nx.compose_all(list(chain([cmos_graph], input_sources)))

    return cmos_graph


def test_synthesize_circuit():
    ''' Test synthesizing an inverter.
    :return:
    '''
    a = sympy.symbols('a')
    cmos = synthesize_circuit(~a)
    assert len(cmos.edges) == 2


def synthesize_minimal_circuit(z: sympy.Symbol) -> nx.MultiGraph:
    """ Synthesize `z` and `Not(z)` + an inverter and return the circuit with less transistors.
    :param z:
    :return:
    """
    z_inv = ~z

    cmos_graph_inv = synthesize_circuit(z_inv)
    cmos_graph = synthesize_circuit(z)

    if len(cmos_graph.edges) > len(cmos_graph_inv.edges) + 2:
        logger.debug('Construct inverted circuit + inverter: Neg(Neg(%s))', z)
        cmos_graph = cmos_graph_inv
        # Append inverter
        output = sympy.symbols('output_inv')
        inverter = _formula_to_cmos_network(~output)

        cmos_graph = nx.relabel_nodes(cmos_graph, {'output': 'output_inv'})
        cmos_graph = nx.compose(cmos_graph, inverter)

    return cmos_graph


def test_synthesize_minimal_circuit():
    a, b = sympy.symbols('a b')
    z = a & b
    cmos = synthesize_minimal_circuit(z)
    assert len(cmos.edges) == 4 + 2


def synthesize_transistors(f: sympy.Symbol) -> List[Transistor]:
    cmos_graph = synthesize_minimal_circuit(f)
    input_signals = set(f.atoms())
    transistors = _cmos_graph_to_transistors(input_signals, cmos_graph)

    return transistors


def formula_from_string(s: str) -> sympy.Symbol:
    return sympy.parsing.sympy_parser.parse_expr(s)


def test_from_minterms():
    """
    Example of how to get a boolean formula from minterms.
    :return:
    """
    w, x, y, z = sympy.symbols('w x y z')

    minterms = [(0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)]
    dontcares = [(0, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 1)]

    # Convert into sum-of-products form.
    sop = SOPform([w, x, y, z], minterms, dontcares)
    # sop = sympy.simplify_logic(sop)

    # Create lambda function from boolean formula
    func = lambdify((w, x, y, z), sop)

    # Check if func really implements the truth table
    for args in product(*([[0, 1]] * 4)):
        if args not in dontcares:
            if args in minterms:
                assert func(*args)
            else:
                assert not func(*args)


def test_generate_all_n_input_formulas():
    n = 2

    inputs = [sympy.Symbol(chr(ord('a') + i)) for i in range(n)]

    truth_table_len = 2 ** n
    num_truth_tables = truth_table_len ** 2

    truth_table_input = list(product(*([[0, 1]] * n)))
    truth_tables_z = product(*([[0, 1]] * truth_table_len))

    for truth_table in truth_tables_z:

        min_terms = []
        for i, z in zip(truth_table_input, truth_table):
            if z == 1:
                min_terms.append(i)

        sop = SOPform(inputs, min_terms, dontcares=[])


def enumerate_all_multi_graphs(num_nodes, num_edges):
    """

    :param num_nodes:
    :param num_edges:
    :return:
    """

    """
    Start with g = one edge, two nodes.
    step: insert edge
        pick two existing nodes to create an edge
        if node limit not yet reached:
            pick a node split connected edges in two groups (2^n-2 possibilities) -> one node each
            connect both nodes with new edge (or pick an edge and split it??)
    """

    def backtrack(graph: nx.MultiGraph, num_nodes, num_edges):
        if num_edges == 0:
            return [graph]

        graphs = []

        new_edge_id = graph.number_of_edges() + 1
        if num_nodes > 0:
            # Create new graph by splitting an edge and inserting a new node.
            new_node_id = graph.number_of_nodes() + 1
            for a, b, key in graph.edges(keys=True):
                g2 = graph.copy()
                g2.remove_edge(a, b, key=key)
                g2.add_edge(a, new_node_id, key)
                g2.add_edge(new_node_id, b, new_edge_id)

                r = backtrack(g2, num_nodes - 1, num_edges - 1)
                graphs.extend(r)

        if num_edges > 0:
            # Add edge by connecting two existing nodes.
            node_combinations = combinations(graph.nodes, 2)
            for a, b in node_combinations:
                if a != b:
                    g2 = graph.copy()
                    g2.add_edge(a, b, new_edge_id)

                    r = backtrack(g2, num_nodes - 0, num_edges - 1)
                    graphs.extend(r)
                else:
                    pass
                    # TODO: Split node and insert edge if num_edges > 0

        return graphs

    # Start with a single edge.
    graph = nx.MultiGraph()
    graph.add_edge('a', 'b', 1)
    # Mark a, b as terminal nodes of the graph (such as vdd and output).
    graph.nodes['a']['terminal'] = 'a'
    graph.nodes['b']['terminal'] = 'b'

    graphs = backtrack(graph, num_nodes, num_edges)

    return graphs


def test_enumerate_all_multigraphs():
    graphs = enumerate_all_multi_graphs(5, 5)

    def node_match(attr1, attr2):
        return attr1.get('terminal', None) == attr2.get('terminal', None)

    # for i, ref in enumerate(graphs):
    #     iso0 = [g for g in graphs\ if nx.is_isomorphic(g, ref, node_match=node_match)]
    #     print('num isomorphic graphs to graph[%d]: %d'%(i, len(iso0)))

    print(len(graphs))
