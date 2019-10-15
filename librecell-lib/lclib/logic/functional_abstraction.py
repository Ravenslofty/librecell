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
from typing import Any, Dict, List, Iterable, Tuple, Set
from enum import Enum
import collections
import sympy
from sympy.logic import simplify_logic, satisfiable
from sympy.logic import boolalg

from lclayout.data_types import ChannelType
import logging

logger = logging.getLogger(__name__)


def bool_equals(f1: boolalg.Boolean, f2: boolalg.Boolean) -> bool:
    """
    Check equality of two boolean formulas.
    :param f1:
    :param f2:
    :return:
    """
    return not satisfiable(f1 ^ f2)


def boolean_derivatives(f: boolalg.Boolean, x: sympy.Symbol) \
        -> Tuple[boolalg.Boolean, boolalg.Boolean, boolalg.Boolean]:
    """
    Calculate the boolean derivative, positive derivative and negative derivative.
    :param f:
    :param x:
    :return: (derivative, positive derivative, negative derivative)
    """
    # TODO: Use this also for is_unate_in_xi in util.
    assert isinstance(x, sympy.Symbol)

    f0 = simplify_logic(f.subs({x: False}))
    f1 = simplify_logic(f.subs({x: True}))
    positive_derivative = simplify_logic(~f0 & f1)
    negative_derivative = simplify_logic(f0 & ~f1)

    derivative = simplify_logic(positive_derivative | negative_derivative)  # == f0 ^ f1
    print('calculate derivative: d/d{} ({})) = {}'.format(x, f, derivative))
    return derivative, positive_derivative, negative_derivative


def _get_gate_nets(graph: nx.MultiGraph) -> Set:
    """
    Return a set of all net names that connect to a transistor gate.
    :param graph:
    :return: Set of net names.
    """
    all_gate_nets = {net_name for (_a, _b, (net_name, _channel_type)) in graph.edges(keys=True)}
    return all_gate_nets


def _find_input_gates(graph: nx.MultiGraph) -> Set:
    """
    Find names of input signals.
    Every net that is connected only to transistor gates is considered an input to the cell.
    :param graph:
    :return: Set of input signal names.
    """

    all_gate_nets = _get_gate_nets(graph)
    all_nodes = set(graph.nodes)
    input_nets = all_gate_nets - all_nodes

    return input_nets


def test_find_input_gates():
    g = nx.MultiGraph()
    g.add_edge('vdd', 'nand', ('a', ChannelType.PMOS))
    g.add_edge('vdd', 'nand', ('b', ChannelType.PMOS))
    g.add_edge('gnd', '1', ('a', ChannelType.NMOS))
    g.add_edge('1', 'nand', ('b', ChannelType.NMOS))
    g.add_edge('vdd', 'output', ('nand', ChannelType.PMOS))
    g.add_edge('gnd', 'output', ('nand', ChannelType.NMOS))

    inputs = _find_input_gates(g)
    assert inputs == {'a', 'b'}


def _all_simple_paths_multigraph(graph: nx.MultiGraph, source, target, cutoff=None):
    """
    Enumerate all simple paths (no node occurs more than once) from source to target.
    Yields edges inclusive keys of all simple paths in a multi graph.
    :param graph: 
    :param source:
    :param target:
    :param cutoff:
    :return: Generator object.
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


def _get_conductivity_conditions(cmos_graph: nx.MultiGraph,
                                 inputs: Set,
                                 output_node) -> Dict[sympy.Symbol, boolalg.Boolean]:
    """
    For each input-output pair find the condition that there is a conductive path.
    :param cmos_graph:
    :param vdd_node: Name of VDD supply node.
    :param gnd_node: Name of GND supply node.
    :param inputs: Set of all input pins including power pins.
    :param output_node:
    :return: sympy.Symbol
    """

    gate_input_nets = _find_input_gates(cmos_graph)
    # Find input nets that also connect to a source or drain, i.e. to a transmission gate.
    transmission_input_pins = inputs - gate_input_nets
    logger.info("Input pins to a transmission gate: {}".format(transmission_input_pins))
    if len(transmission_input_pins) < 2:
        logger.warning("`inputs` is expected to also contain VDD and GND.")

    def conductivity_condition(cmos_graph: nx.MultiGraph, source, target) -> boolalg.Boolean:
        """
        Find a boolean equation that evaluates to true iff there is a conductive path from `source` to `target`
        given the input signals.
        :param cmos_graph:
        :param source:
        :param target:
        :return: Boolean function. (sympy.Symbol)
        """
        # Get all simple paths from source to target.
        all_paths = list(_all_simple_paths_multigraph(cmos_graph, source, target))
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
        # Try to simplify the boolean expression.
        f = simplify_logic(f)
        logger.debug("Conductivity condition from '{}' to '{}': {}".format(source, target, f))
        return f

    def remove_nodes(graph: nx.MultiGraph, delete_nodes: Set) -> nx.MultiGraph:
        remaining_nodes = graph.node - delete_nodes
        return graph.subgraph(remaining_nodes)

    # Calculate conductivity conditions from each input-pin (i.e. power pins and inputs to transmission gates) to output.
    conductivity_conditions = {
        sympy.Symbol(i):
            conductivity_condition(
                # Remove all other nodes connected to transmission input pins.
                # This avoids for instance finding a path to GND leading over VDD.
                remove_nodes(cmos_graph, transmission_input_pins - {i}),
                output_node,
                i
            )
        for i in transmission_input_pins
    }

    return conductivity_conditions


#
# def _cmos_graph_to_formula(cmos_graph: nx.MultiGraph,
#                            vdd_node,
#                            gnd_node,
#                            output_node,
#                            input_pins: Set = None) -> boolalg.Boolean:
#     """
#     Find the boolean formula implemented by the push-pull network `cmos_graph`.
#     :param cmos_graph:
#     :param vdd_node: Name of VDD supply node.
#     :param gnd_node: Name of GND supply node.
#     :param input_pins: Set of all input pins including power pins.
#     :param output_node:
#     :return: sympy.Symbol
#     """
#
#     if input_pins is None:
#         input_pins = set()
#     else:
#         input_pins = input_pins.copy()
#
#     input_pins.add(vdd_node)
#     input_pins.add(gnd_node)
#
#     logger.debug("Input nets: {}".format(input_pins))
#     logger.debug("Output net: {}".format(output_node))
#
#     # Calculate conductivity conditions from each input-pin to output.
#     conductivity_conditions = _get_conductivity_conditions(cmos_graph, input_pins, output_node)
#
#     # Find condition that output is connected to VDD.
#     output_at_vdd = conductivity_conditions[sympy.Symbol(vdd_node)]
#     # # Find condition that output is connected to GND.
#     # output_at_gnd = conductivity_conditions[gnd_node]
#     #
#     # # Check if the two conditions are complementary.
#     # is_complementary = _bool_equals(output_at_gnd, ~output_at_vdd)
#     #
#     # # Check if it is possible to create a path connecting VDD and GND.
#     # short_condition = output_at_vdd & output_at_gnd
#     # has_short = satisfiable(short_condition)
#     #
#     # # Check if it is possible to disconnect the output from both VDD and GND (high-impedance).
#     # tri_state_condition = simplify_logic((~output_at_vdd) & (~output_at_gnd))
#     # has_tri_state = satisfiable(tri_state_condition)
#
#     # TODO: This only works if the circuit is complementary.
#     f_out = simplify_logic(output_at_vdd)
#     # logger.info("Deduced formula: {} = {}".format(output_node, f_out))
#     # logger.info("Is complementary circuit: {}".format(is_complementary))
#     # assert is_complementary, "Non-complementary circuits not supported yet."
#     #
#     # logger.info("Has tri-state: {}".format(has_tri_state))
#     # if has_tri_state:
#     #     logger.info("High impedance output when: {}".format(tri_state_condition))
#     #
#     # logger.info("Has short circuit: {}".format(has_short))
#     # if has_short:
#     #     logger.warning("Short circuit when: {}".format(short_condition))
#
#     return f_out


def _high_impedance_condition(conductivity_conditions: Dict[sympy.Symbol, boolalg.Boolean]) -> boolalg.Boolean:
    pass


def _is_complementary(f1: boolalg.Boolean, f2: boolalg.Boolean) -> bool:
    """
    Check if two formulas are complementary.
    :param f1:
    :param f2:
    :return:
    """
    return bool_equals(f1, ~f2)


def test_cmos_graph_to_formula():
    # Create CMOS network of a nand gate and check if `evaluate_cmos_graph` behaves like the nand function.
    g = nx.MultiGraph()
    g.add_edge('vdd', 'output', ('a', ChannelType.PMOS))
    g.add_edge('vdd', 'output', ('b', ChannelType.PMOS))
    g.add_edge('gnd', '1', ('a', ChannelType.NMOS))
    g.add_edge('1', 'output', ('b', ChannelType.NMOS))

    formula = _cmos_graph_to_formula(g, 'vdd', 'gnd', 'output')

    # Verify that the deduced formula equals a NAND.
    a, b = sympy.symbols('a b')
    nand = ~(a & b)
    assert formula.equals(nand), "Transformation of CMOS graph into formula failed."


def complex_cmos_graph_to_formula(cmos_graph: nx.MultiGraph,
                                  output_nodes: Set,
                                  input_pins: Set) \
        -> Tuple[Dict[Any, Dict[sympy.Symbol, boolalg.Boolean]], set]:
    """
    Iteratively find formulas of intermediate nodes in complex gates which consist of multiple pull-up/pull-down networks.
    :param cmos_graph:
    :param output_nodes:
    :param input_pins: Specify additional input pins that could otherwise not be deduced.
        This is mainly meant for power supply pins and inputs that connect not only to transistor gates but also source or drain.
    :return: (Dict[intermediate variable, expression for it], Set[input variables of the circuit])
    """

    for n in output_nodes:
        assert n in cmos_graph.node, "Output node is not in the graph: {}".format(n)

    assert len(input_pins) >= 2, "`input_pins` must at least contain 2 nodes (GND, VDD). Found: {}".format(input_pins)
    input_pins = input_pins.copy()

    # Consider all nodes as input variables that are either connected to transistor
    # gates only or are specified by the caller.
    deduced_input_pins = _find_input_gates(cmos_graph)
    logger.info("Deduced input pins: {}".format(deduced_input_pins))
    inputs = deduced_input_pins | input_pins
    logger.info("All input pins: {}".format(deduced_input_pins))

    unknown_nodes = {n for n in output_nodes}
    known_nodes = {i for i in inputs}
    output_formulas = dict()

    # Loop as long as there is a intermediary node with unknown value.
    while unknown_nodes:
        # Grab a node with unknown boolean expression.
        temp_output_node = unknown_nodes.pop()
        assert temp_output_node not in known_nodes
        logger.debug("Find conductivity conditions for: {}".format(temp_output_node))
        conductivity_conditions = _get_conductivity_conditions(cmos_graph, input_pins, temp_output_node)

        output_formulas[temp_output_node] = conductivity_conditions
        known_nodes.add(temp_output_node)

        # Find all variables that occur in the conductivity conditions to later resolve them too.
        inputs_to_f = {a.name for f in conductivity_conditions.values() for a in f.atoms()}

        # Update the set of unknown variables.
        unknown_nodes = (unknown_nodes | inputs_to_f) - known_nodes

    return output_formulas, inputs


def test_complex_cmos_graph_to_formula():
    # Create CMOS network of a AND gate (NAND -> INV).
    g = nx.MultiGraph()
    g.add_edge('vdd', 'nand', ('a', ChannelType.PMOS))
    g.add_edge('vdd', 'nand', ('b', ChannelType.PMOS))
    g.add_edge('gnd', '1', ('a', ChannelType.NMOS))
    g.add_edge('1', 'nand', ('b', ChannelType.NMOS))

    g.add_edge('vdd', 'output', ('nand', ChannelType.PMOS))
    g.add_edge('gnd', 'output', ('nand', ChannelType.NMOS))

    conductivity_conditions, inputs = complex_cmos_graph_to_formula(g, output_nodes={'output'},
                                                                    input_pins={'vdd', 'gnd'})
    print(conductivity_conditions)
    formulas = {sympy.Symbol(output): cc[sympy.Symbol('vdd')] for output, cc in conductivity_conditions.items()}

    # Convert from strings into sympy symbols.
    inputs = {sympy.Symbol(i) for i in inputs}
    print('formulas: ', formulas)
    print('inputs = ', inputs)

    # Detect loops in the circuit.
    # Create a graph representing the dependencies of the variables/expressions.
    dependency_graph = nx.DiGraph()
    for atom, expression in formulas.items():
        dependency_graph.add_edge(atom, expression)

    # Check for cycles.
    cycles = list(nx.simple_cycles(dependency_graph))

    assert len(cycles) == 0, "Abstraction of feed-back loops not yet supported."

    print('cycles = ', cycles)

    def resolve_intermediate_variables(formulas: Dict[sympy.Symbol, sympy.Symbol],
                                       output: sympy.Symbol) -> boolalg.Boolean:
        f = formulas[output].copy()
        # TODO: detect loops
        while f.atoms() - inputs:
            f = f.subs(formulas)
            f = simplify_logic(f)
        return f

    # Solve equation system for output.
    f = resolve_intermediate_variables(formulas, sympy.Symbol('output'))

    f = simplify_logic(f)
    print('f = ', f)

    # Verify that the deduced formula equals a NAND.
    a, b = sympy.symbols('a b')
    AND = (a & b)
    assert f.equals(AND), "Transformation of CMOS graph into formula failed."


def analyze_circuit_graph(graph: nx.MultiGraph,
                          pins_of_interest: Set,
                          constant_input_pins: Dict[Any, bool] = None,
                          user_input_nets: Set = None
                          ) -> Dict[sympy.Symbol, boolalg.Boolean]:
    """


    :param graph:
    :type graph: nx.MultiGraph
    :param pins_of_interest:
    :param constant_input_pins: Define input pins that have a constant and known value such as `{'vdd': True, 'gnd': False}`.
    :type constant_input_pins: Dict[Any, bool]
    :param user_input_nets: A set of input pin names.
        Some inputs cannot automatically be found and must be provided by the user.
        Input nets that connect not only to transistor gates but also source or drains need to be specified manually.
        This can happen for cells containing transmission gates.
    :return: Dict['output pin', ]
    """
    pins_of_interest = set(pins_of_interest)

    if constant_input_pins is None:
        constant_input_pins = dict()

    gate_nets = _get_gate_nets(graph)
    nets = set(graph.nodes)
    all_nets = gate_nets | nets

    # Sanity check for the inputs.
    for p in pins_of_interest:
        assert p in all_nets, "Net '{}' is not in the graph.".format(p)

    logger.info("VDD nets: {}".format(", ".join([p for p, v in constant_input_pins.items() if v])))
    logger.info("GND nets: {}".format(", ".join([p for p, v in constant_input_pins.items() if not v])))

    if user_input_nets is None:
        user_input_nets = set()
    else:
        user_input_nets = set(user_input_nets)

    input_nets = user_input_nets | _find_input_gates(graph) | constant_input_pins.keys()
    output_nodes = pins_of_interest - input_nets
    logger.info("Detected input nets: {}".format(input_nets))
    logger.info("Detected output nets: {}".format(output_nodes))

    conductivity_conditions, inputs = complex_cmos_graph_to_formula(graph,
                                                                    input_pins=input_nets,
                                                                    output_nodes=output_nodes,
                                                                    )
    print('conductivity_conditions = ', conductivity_conditions)

    # Convert keys into symbols.
    conductivity_conditions = {sympy.Symbol(k): v for k, v in conductivity_conditions.items()}

    formulas = dict()
    for output, cc in conductivity_conditions.items():
        or_terms = []
        for input_pin, condition in cc.items():
            assert isinstance(input_pin, sympy.Symbol)
            assert isinstance(condition, boolalg.Boolean)
            # if `condition` then output is `connected` to `input_pin`.
            connected_to_input = condition & input_pin
            connected_to_input = simplify_logic(connected_to_input)
            or_terms.append(connected_to_input)

        # TODO: if more than one of the OR terms evaluates to True, then there could be a short circuit.

        # Logic OR of all elements.
        or_formula = sympy.Or(*or_terms)
        formulas[output] = or_formula

    # Add known values for VDD, GND
    constants = {sympy.Symbol(k): v for k, v in constant_input_pins.items()}

    # Simplify formulas by substituting VDD and GND with known values.
    formulas = {k: simplify_logic(f.subs(constants)) for k, f in formulas.items()}
    logger.debug('formulas = {}'.format(formulas))

    # Convert from strings into sympy symbols.
    inputs = {sympy.Symbol(i) for i in inputs} - set(constants.keys())
    logger.debug('inputs = {}'.format(inputs))

    # Detect loops in the circuit.
    # Create a graph representing the dependencies of the variables/expressions.
    dependency_graph = nx.DiGraph()
    for output, expression in formulas.items():
        if isinstance(expression, boolalg.Boolean):
            for atom in expression.atoms():
                # Output depends on every variable (atom) in the expression.
                dependency_graph.add_edge(output, atom)
        elif isinstance(expression, bool):
            # Handle True and False constants.
            dependency_graph.add_edge(output, expression)
        else:
            assert False, "Type not supported: '{}'".format(type(expression))

    # import matplotlib.pyplot as plt
    # nx.draw_networkx(dependency_graph)
    # plt.draw()
    # plt.show()

    # Check for cycles.
    cycles = list(nx.simple_cycles(dependency_graph))
    logger.info("Number of feed-back loops: {}".format(len(cycles)))

    for cycle in cycles:
        logger.info("cycle: {}".format(cycle))
        print('cycle = ', cycle)
        for el in cycle:
            print('--> {} = {}'.format(el, formulas[el]))

    # assert len(cycles) == 0, "Abstraction of feed-back loops is not yet supported."

    # Collect all nets that belong to a memory cycle.
    nets_of_memory_cycles = {n for c in cycles for n in c}

    class Memory:
        def __init__(self, data: boolalg.Boolean, write_condition: boolalg.Boolean,
                     oscillation_condition: boolalg.Boolean):
            self.data = data
            self.write_condition = write_condition
            self.oscillation_condition = oscillation_condition

        def __str__(self):
            return "Memory({}, write = {})".format(self.data, self.write_condition)

        def __repr__(self):
            return str(self)

    print('cycles = ', cycles)

    def resolve_intermediate_variables(formulas: Dict[sympy.Symbol, boolalg.Boolean],
                                       inputs: Set[sympy.Symbol],
                                       root: boolalg.Boolean,
                                       break_on_loop: bool = False):
        f = formulas[root]
        f = simplify_logic(f)
        # Remember previous results to be able to detect loops.
        previous_formulas = {f}
        while f.atoms() - inputs:
            f = f.subs(formulas)
            f = simplify_logic(f)

            if f in previous_formulas:
                if break_on_loop:
                    print('loop detected')
                    break
                else:
                    assert False, "Equation system contains a loop!"
            previous_formulas.add(f)

        return f

    def _analyze_cycle(cycle: List[sympy.Symbol], cycle_output: sympy.Symbol, formulas):

        # Break the loop by treating cycle_start as an unknown.
        # _formulas = {k: v for k, v in formulas.items() if k in cycle}
        cycle_start_resolved = resolve_intermediate_variables(formulas, inputs, cycle_output, break_on_loop=True)

        d, dp, dn = boolean_derivatives(cycle_start_resolved, cycle_output)

        write_condition = ~dp
        oscillation_condition = dn
        print('write_condition = ', write_condition)
        print('oscillation_condition = ', oscillation_condition)

    def derive_memory(memory_output_net: sympy.Symbol, inputs: Set[sympy.Symbol], formulas) -> Memory:
        print("derive memory from: {}".format(memory_output_net))
        memory_output_resolved = resolve_intermediate_variables(formulas, inputs, memory_output_net, break_on_loop=True)
        d, dp, dn = boolean_derivatives(memory_output_resolved, memory_output_net)
        write_condition = ~dp
        oscillation_condition = dn

        # Find an expression for the memory output when the write condition is met.
        write_condition_model = satisfiable(write_condition)
        if not write_condition_model:
            logger.warning("Detected a memory loop that is not possible to write to.")
        data = simplify_logic(memory_output_resolved.subs(write_condition_model))

        return Memory(data=data, write_condition=write_condition, oscillation_condition=oscillation_condition)

    # # Analyze potential memory elements.
    # for cycle in cycles:
    #     assert len(cycle) >= 2, "Cycle is expected to have a length longer than 1."
    #     # TODO: cycle_end should be the output of the memory, cycle_start the input.
    #     # TODO: 1. find memory cycles, 2. trace back output until input pins or memory, 3. resolve memory, ... (Use some sort of task queue.)
    #     # memory_outputs = [c for c in cycle if
    #     #
    #     #                   ]
    #     memory_output = cycle[0]
    #
    #     _analyze_cycle(cycle, memory_output, formulas)

    # Solve equation system for output.
    # TODO: stop resolving at memory elements.
    output_formulas = dict()
    memory_nets = dict()

    wavefront = {sympy.Symbol(o) for o in output_nodes}
    while wavefront:

        out = wavefront.pop()
        assert out not in output_formulas
        assert out not in memory_nets

        new_wavefront = set()

        if out not in nets_of_memory_cycles:
            print("find formula for", out)
            assert isinstance(out, sympy.Symbol)
            formula = resolve_intermediate_variables(formulas, inputs | nets_of_memory_cycles, out)
            output_formulas[out] = formula

            # Find new wavefront.
            atoms = formula.atoms()
            new_wavefront.update(atoms)

            # Find inputs into formula that come from a memory.
            new_atoms = atoms - output_formulas.keys() - memory_nets.keys() - inputs
            unknown_memory_nets = new_atoms & nets_of_memory_cycles
            print("unknown inputs into", out, "from memory:", unknown_memory_nets)
        else:
            # Resolve memory.
            memory_net = out
            _this_memory_cycle = [c for c in cycles if memory_net in c]
            assert len(_this_memory_cycle) == 1
            _this_memory_cycle = set(_this_memory_cycle[0])

            _inputs = inputs | nets_of_memory_cycles - _this_memory_cycle
            memory = derive_memory(memory_net, _inputs, formulas)

            memory_nets[memory_net] = memory

            # Find new wavefront.
            atoms = memory.write_condition.atoms() | memory.data.atoms()
            new_wavefront.update(atoms)

        wavefront.update(new_wavefront - output_formulas.keys() - memory_nets.keys() - inputs)

    # Simplify formulas in memories.
    for m in memory_nets.values():
        m.data = simplify_logic(m.data.subs(output_formulas))
        m.write_condition = simplify_logic(m.write_condition.subs(output_formulas))
        m.oscillation_condition = simplify_logic(m.oscillation_condition.subs(output_formulas))

    print("Formulas: {}".format(output_formulas))
    print("Memories: {}".format(memory_nets))
    assert len(cycles) == 0, "Abstraction of feed-back loops is not yet supported."
    return output_formulas


def test_analyze_circuit_graph():
    # Create CMOS network of a AND gate (NAND -> INV).
    g = nx.MultiGraph()
    g.add_edge('vdd', 'nand', ('a', ChannelType.PMOS))
    g.add_edge('vdd', 'nand', ('b', ChannelType.PMOS))
    g.add_edge('gnd', '1', ('a', ChannelType.NMOS))
    g.add_edge('1', 'nand', ('b', ChannelType.NMOS))
    g.add_edge('vdd', 'output', ('nand', ChannelType.PMOS))
    g.add_edge('gnd', 'output', ('nand', ChannelType.NMOS))

    pins_of_interest = {'output'}
    known_pins = {'vdd': True, 'gnd': False}
    result = analyze_circuit_graph(g, pins_of_interest=pins_of_interest, constant_input_pins=known_pins)

    # Verify that the deduced boolean function is equal to the AND function.
    a, b = sympy.symbols('a, b')
    assert bool_equals(result[sympy.Symbol('output')], a & b)


def test_analyze_circuit_graph_transmission_gate_xor():
    g = nx.MultiGraph()
    # Represent an XOR gate with transmission-gates in its graph form.
    # Edges correspond to transistors, nodes to nets. The 'key' of the edge
    # Contains a Tuple ('net name of the gate', 'channel type').

    # Inverter A
    g.add_edge('vdd', 'a_not', ('a', ChannelType.PMOS))
    g.add_edge('gnd', 'a_not', ('a', ChannelType.NMOS))
    # Inverter B
    g.add_edge('vdd', 'b_not', ('b', ChannelType.PMOS))
    g.add_edge('gnd', 'b_not', ('b', ChannelType.NMOS))
    # Transmission gates
    g.add_edge('a_not', 'c', ('b', ChannelType.NMOS))
    g.add_edge('a_not', 'c', ('b_not', ChannelType.PMOS))
    g.add_edge('a', 'c', ('b_not', ChannelType.NMOS))
    g.add_edge('a', 'c', ('b', ChannelType.PMOS))

    # Pin 'a' must be explicitly given as an input pin.
    # It connects not only to gates but also to source/drains of transistors
    # and therefore cannot be deduced to be an input pin.
    pins_of_interest = {'a', 'b', 'c'}
    known_pins = {'vdd': True, 'gnd': False}
    result = analyze_circuit_graph(g, pins_of_interest=pins_of_interest, constant_input_pins=known_pins,
                                   user_input_nets={'a', 'b'})

    # Verify that the deduced boolean function is equal to the XOR function.
    a, b = sympy.symbols('a, b')
    assert bool_equals(result[sympy.Symbol('c')], a ^ b)


def test_analyze_circuit_graph_latch():
    g = nx.MultiGraph()
    # Network of a LATCH.
    edges = [
        # Command for converting a raw SPICE transistor netlist into this format:
        # cat raw_netlist.sp | grep -v + | awk '{ print "(\"" $2 "\", \"" $4 "\", (\"" $3  "\", " $6 ")),"}' \
        # | sed 's/pmos/ChannelType.PMOS/g' | sed 's/nmos/ChannelType.NMOS/g'
        ("vdd", "a_2_6#", ("CLK", ChannelType.PMOS)),
        ("a_18_74#", "vdd", ("D", ChannelType.PMOS)),
        ("a_23_6#", "a_18_74#", ("a_2_6#", ChannelType.PMOS)),
        ("a_35_84#", "a_23_6#", ("CLK", ChannelType.PMOS)),
        ("vdd", "a_35_84#", ("Q", ChannelType.PMOS)),
        ("gnd", "a_2_6#", ("CLK", ChannelType.NMOS)),
        ("Q", "vdd", ("a_23_6#", ChannelType.PMOS)),
        ("a_18_6#", "gnd", ("D", ChannelType.NMOS)),
        ("a_23_6#", "a_18_6#", ("CLK", ChannelType.NMOS)),
        ("a_35_6#", "a_23_6#", ("a_2_6#", ChannelType.NMOS)),
        ("gnd", "a_35_6#", ("Q", ChannelType.NMOS)),
        ("Q", "gnd", ("a_23_6#", ChannelType.NMOS)),
    ]
    for e in edges:
        g.add_edge(*e)

    pins_of_interest = {'CLK', 'D', 'Q'}
    known_pins = {'vdd': True, 'gnd': False}
    result = analyze_circuit_graph(g, pins_of_interest=pins_of_interest, constant_input_pins=known_pins)
