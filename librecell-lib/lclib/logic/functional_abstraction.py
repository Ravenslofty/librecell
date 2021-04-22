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

import itertools
import networkx as nx
from typing import Any, Dict, List, Iterable, Tuple, Set
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
Extract logic formulas and memory loops from a transistor-level circuit graph.
"""


class CombinationalOutput:
    """
    Description of an output signal of a combinatorial circuit.
    """

    def __init__(self, function: boolalg.Boolean, high_impedance: boolalg.Boolean):
        self.function = function
        self.high_impedance = high_impedance

    def is_tristate(self):
        """
        Check if the output have be high-impedance.
        Check if the high-impedance condition is satisfiable.
        :return: bool
        """
        return satisfiable(self.high_impedance)

    def __str__(self):
        return "CombinationalOutput(f = {}, Z = {})".format(self.function, self.high_impedance)

    def __repr__(self):
        return str(self)


class Memory:
    """
    Data structure for a memory loop (latch).
    """

    def __init__(self,
                 data: boolalg.Boolean,
                 write_condition: boolalg.Boolean,
                 oscillation_condition: boolalg.Boolean):
        """

        :param data: The input data signal.
        :param write_condition: The condition that makes this loop transparent for new input data.
        When the write condition is `False` then the loop stores the input data.
        :param oscillation_condition: When the oscillation condition is met the loop oscillates or causes a short circuit.
        """
        self.data = data
        self.write_condition = write_condition
        self.oscillation_condition = oscillation_condition
        # ID of the memory loop.
        self.loop_id = None

    def __str__(self):
        return f"Memory(data = {self.data}, write = {self.write_condition}, " \
               f"loop_id = {self.loop_id}, oscillate = {self.oscillation_condition})"

    def __repr__(self):
        return str(self)


class AbstractCircuit:
    """
    Describe the circuit as a set of boolean formulas and latches.
    """

    def __init__(self,
                 output_pins: Set[sympy.Symbol],
                 output_formulas: Dict[boolalg.BooleanAtom, CombinationalOutput],
                 latches: Dict[boolalg.BooleanAtom, Memory]):
        """
        Describe the circuit as a set of boolean formulas and latches.
        Holds the result of `analyze_circuit_graph`.

        :param output_pins: Set of all output pins.
        :param output_formulas: Dictionary that maps output symbols to their boolean formulas.
        :param latches: Dictionary that holds the memory objects for each signal that is driven by a memory/latch.
        """
        self.output_pins = output_pins
        self.outputs = output_formulas
        self.latches = latches

    def is_sequential(self) -> bool:
        """
        Test if there are latches in this circuit.
        :return:
        """
        return len(self.latches) > 0


def simplify_logic(f: boolalg.Boolean, force: bool = True) -> boolalg.Boolean:
    """
    Simplify a boolean formula.
    :param f:
    :param force:
    :return:
    """
    logger.debug(f"Simplify formula: {f}")
    return sympy_simplify_logic(f, force=force)


def simplify_with_assumption(assumption: boolalg.Boolean, formula: boolalg.Boolean) -> boolalg.Boolean:
    """
    Simplify a formula while assuming that `assumption` is satisfied.
    The simplification might be incorrect when the assumption is not satisfied.
    In other words all variable assignments that invalidate the assumption are treated as don't-cares.
    :param assumption:
    :param formula: The formula to be simplified.
    :return: Simplified formula under the given assumption.
    """

    # Extract variables.
    all_vars = set()
    all_vars.update(assumption.atoms())
    all_vars.update(formula.atoms())
    all_vars = [v for v in all_vars if v is not boolalg.true and v is not boolalg.false]
    all_vars = sorted(all_vars, key=lambda a: a.name)

    # Create a truth table.
    # Treat all rows that don't satisfy the assumption as "don't care".
    truth_table_assumption = boolalg.truth_table(assumption, all_vars)
    truth_table_formula = boolalg.truth_table(formula, all_vars, input=False)

    minterms = []
    dont_cares = []
    for (inputs, value_assumption), value_formula in zip(truth_table_assumption, truth_table_formula):

        if not value_assumption:
            dont_cares.append(inputs)
        elif value_formula:
            minterms.append(inputs)

    f = boolalg.SOPform(variables=all_vars, minterms=minterms, dontcares=dont_cares)

    return f


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
    logger.debug('calculate derivative: d/d{} ({})) = {}'.format(x, f, derivative))
    logger.debug('calculate pos. derivative: d+/d{} ({})) = {}'.format(x, f, positive_derivative))
    logger.debug('calculate neg. derivative: d-/d{} ({})) = {}'.format(x, f, negative_derivative))
    return derivative, positive_derivative, negative_derivative


def _get_gate_nets(graph: nx.MultiGraph) -> Set:
    """
    Return a set of all net names that connect to a transistor gate.
    Gate net names are stored as keys of the edges in the multi-graph.
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
    :param source: Source node.
    :param target: Target node.
    :param cutoff: Stopping condition. Maximal length of the path.
    :return: Generator object.
    """

    # Sanity check: source and target node must be in the graph.
    if source not in graph:
        raise nx.NodeNotFound('source node %s not in graph' % source)
    if target not in graph:
        raise nx.NodeNotFound('target node %s not in graph' % target)

    # Trivial solution.
    if source == target:
        return []

    # Set the cutoff to the number of nodes in the graph minus 1.
    if cutoff is None:
        cutoff = len(graph) - 1

    # If the cutoff is 0, return an empty list.
    if cutoff < 1:
        return []

    # Set of nodes that have been visited already.
    # An ordered dict is used such that nodes can be removed from the set
    # according to the reversed insertion order.
    # This works basically as a stack but allows for faster membership test than a list.
    visited = collections.OrderedDict.fromkeys([source])

    edges = list()  # Store sequence of edges.

    # Put on the stack an iterator over all edges adjacent to the source node.
    stack = [
        (((u, v), key) for u, v, key in graph.edges(source, keys=True))
    ]
    while stack:
        children = stack[-1]  # Take the top of the stack.
        # Take the next element of the iterator.
        (child_source, child), child_key = next(children, ((None, None), None))
        if child is None:
            # The iterator on top of the stack is empty.
            stack.pop()  # Drop the empty iterator.
            # Rewind to the previous 'recursion' level.
            visited.popitem()
            edges = edges[:-1]
        elif len(visited) < cutoff:
            if child == target:
                # Target has been found.
                yield list(edges) + [((child_source, child), child_key)]
            elif child not in visited:
                # Child node is not yet in the current path.
                visited[child] = None  # Mark as visited.
                # Append child node to the current path.
                edges.append(((child_source, child), child_key))
                # Put on the stack an iterator over all edges adjacent to the child node.
                stack.append(
                    (((u, v), k) for u, v, k in graph.edges(child, keys=True))
                )
        else:
            assert len(visited) == cutoff
            # TODO: Is this correct?
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

    For example an inverter with input A and output Y could be represented as a multi-graph:
    `VDD--[A]--Y--[A]--GND`

    The only output node would be 'Y', inputs would be the set {'A'}.
    The returned dictionary should then contain the condition that there is a conductive path from
    Y to VDD (A == 0) as well as the condition that there is a conductive path from Y to GND (A == 1).


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
        remaining_nodes = graph.nodes() - delete_nodes
        return graph.subgraph(remaining_nodes)

    # Calculate conductivity conditions from each input-pin
    # (i.e. power pins and inputs to transmission gates) to output.
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


def _high_impedance_condition(conductivity_conditions: Dict[sympy.Symbol, boolalg.Boolean]) -> boolalg.Boolean:
    pass


def _is_complementary(f1: boolalg.Boolean, f2: boolalg.Boolean) -> bool:
    """
    Check if two formulas are complementary.
    :param f1:
    :param f2:
    :return: Returns true iff `f1` and `f1` are complementary.
    """
    return bool_equals(f1, ~f2)


def complex_cmos_graph_to_formula(cmos_graph: nx.MultiGraph,
                                  output_nodes: Set,
                                  input_pins: Set) \
        -> Tuple[Dict[Any, Dict[sympy.Symbol, boolalg.Boolean]], set]:
    """
    Iteratively find formulas of intermediate nodes in complex gates which consist of multiple pull-up/pull-down networks.
    :param cmos_graph: The CMOS transistor network represented as a multi-graph.
    :param output_nodes: Set of output nodes. This corresponds to the output nets/pins of the cell.
    :param input_pins: Specify additional input pins that could otherwise not be deduced.
        This is mainly meant for power supply pins and inputs that connect not only to transistor gates but also source or drain.
    :return: (Dict[intermediate variable, expression for it], Set[input variables of the circuit])
    """

    # Sanity check: all output nodes must be in the graph.
    for n in output_nodes:
        assert n in cmos_graph.nodes(), "Output node is not in the graph: {}".format(n)

    # Sanity check: There must be at least two input pins.
    assert len(input_pins) >= 2, "`input_pins` must at least contain 2 nodes (GND, VDD). Found: {}".format(input_pins)
    # Make an independent copy that can be modified without causing side effects.
    input_pins = input_pins.copy()

    # Consider all nodes as input variables that are either connected to transistor
    # gates only or are specified by the caller.
    deduced_input_pins = _find_input_gates(cmos_graph)
    logger.info("Deduced input pins: {}".format(deduced_input_pins))
    inputs = deduced_input_pins | input_pins
    logger.info("All input pins: {}".format(deduced_input_pins))

    # Set of nodes where there is no boolean formula known yet.
    unknown_nodes = {n for n in output_nodes}
    # Set of nodes for which a boolean expression is known.
    known_nodes = {i for i in inputs}
    # Boolean formulas for the outputs.
    output_formulas = dict()

    # Loop as long as there is a intermediary node with unknown value.
    while unknown_nodes:
        # Grab a node with unknown boolean expression.
        temp_output_node = unknown_nodes.pop()
        assert temp_output_node not in known_nodes
        # Follow the pull-up and pull-down paths to find the boolean conditions that
        # tell when there is a conductive path to nodes like VDD, GND (or inputs and internal nodes in case
        # of transmission gates).
        logger.debug("Find conductivity conditions for: {}".format(temp_output_node))
        conductivity_conditions = _get_conductivity_conditions(cmos_graph, input_pins, temp_output_node)

        output_formulas[temp_output_node] = conductivity_conditions
        known_nodes.add(temp_output_node)

        # Find all variables that occur in the conductivity conditions to later resolve them too.
        inputs_to_f = {a.name
                       for f in conductivity_conditions.values()
                       for a in f.atoms(sympy.Symbol)
                       }

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
    logger.debug(f"Conductivity conditions: \n{conductivity_conditions}")
    formulas = {sympy.Symbol(output): cc[sympy.Symbol('vdd')] for output, cc in conductivity_conditions.items()}

    # Convert from strings into sympy symbols.
    inputs = {sympy.Symbol(i) for i in inputs}
    logger.debug(f'formulas = {formulas}')
    logger.debug(f'inputs = {inputs}')

    def resolve_intermediate_variables(formulas: Dict[sympy.Symbol, sympy.Symbol],
                                       output: sympy.Symbol) -> boolalg.Boolean:
        f = formulas[output].copy()
        # TODO: detect loops
        while f.atoms(sympy.Symbol) - inputs:
            f = f.subs(formulas)
            f = simplify_logic(f)
        return f

    # Solve equation system for output.
    f = resolve_intermediate_variables(formulas, sympy.Symbol('output'))

    f = simplify_logic(f)

    # Verify that the deduced formula equals a NAND.
    a, b = sympy.symbols('a b')
    AND = (a & b)
    assert f.equals(AND), "Transformation of CMOS graph into formula failed."


def _resolve_intermediate_variables(formulas: Dict[sympy.Symbol, boolalg.Boolean],
                                    inputs: Set[sympy.Symbol],
                                    formula: boolalg.Boolean) -> boolalg.Boolean:
    """
    Simplify the formula `formula` by iterative substitution with the `formulas`.
    :param formulas: Formulas used for substitution as a dict.
    :param inputs: Define input atoms.
    :param formula: The start.
    :return: Return the formula with all substitutions applied.
    """

    # Set of symbols that will not be substituted.
    stop_atoms = set(inputs)

    while formula.atoms(sympy.Symbol) - stop_atoms:

        for a in formula.atoms(sympy.Symbol) - stop_atoms:
            stop_atoms.update(formula.atoms(sympy.Symbol))
            if a in formulas:
                formula = formula.subs({a: formulas[a]})
                formula = simplify_logic(formula)

    return formula


def test_resolve_intermediate_variables():
    a, b, c = sympy.symbols('a b c')
    formulas = {a: b ^ c, c: ~a}
    inputs = {b}
    r = _resolve_intermediate_variables(formulas, inputs, a)
    assert bool_equals(r, b ^ ~a)


def _formula_dependency_graph(formulas: Dict[sympy.Symbol, boolalg.Boolean]) -> nx.DiGraph:
    """
    Create a graph representing the dependencies of the variables/expressions.
    Used to detect feedback loops.
    :param formulas:
    :return:
    """
    dependency_graph = nx.DiGraph()
    for output, expression in formulas.items():
        if isinstance(expression, boolalg.Boolean):
            for atom in expression.atoms(sympy.Symbol):
                # Output depends on every variable (atom) in the expression.
                dependency_graph.add_edge(output, atom)
        elif isinstance(expression, bool):
            # Handle True and False constants.
            dependency_graph.add_edge(output, expression)
        else:
            assert False, "Type not supported: '{}'".format(type(expression))
    return dependency_graph


def analyze_circuit_graph(graph: nx.MultiGraph,
                          pins_of_interest: Set,
                          constant_input_pins: Dict[Any, bool] = None,
                          user_input_nets: Set = None
                          ) -> AbstractCircuit:
    """
    Analyze a CMOS transistor network and find boolean expressions for the output signals of the `pins_of_interest`.

    The transistor network is represented as a multi-graph, where nodes represent nets and edges represent transistors.
    The end points of an edge correspond to drain and source of the transistor. The label of the edge corresponds to the
    gate net of the transistor.

    :param graph:
    :type graph: nx.MultiGraph
    :param pins_of_interest:
    :param constant_input_pins: Define input pins that have a constant and known value such as `{'vdd': True, 'gnd': False}`.
    :type constant_input_pins: Dict[Any, bool]
    :param user_input_nets: A set of input pin names.
        Some inputs cannot automatically be found and must be provided by the user.
        Input nets that connect not only to transistor gates but also source or drains need to be specified manually.
        This can happen for cells containing transmission gates.
    :return: (Dict['output pin', boolean formula], Dict['output pin', Memory])
    """
    #
    # import matplotlib.pyplot as plt
    # nx.draw_networkx(graph)
    # plt.draw()
    # plt.show()

    pins_of_interest = set(pins_of_interest)

    if constant_input_pins is None:
        constant_input_pins = dict()

    gate_nets = _get_gate_nets(graph)
    nets = set(graph.nodes())
    all_nets = gate_nets | nets

    # Sanity check for the inputs.
    non_present_pins = set()
    for p in pins_of_interest:
        if p not in all_nets:
            logger.warning(f"Net '{p}' is not connected to any transistor.")
            non_present_pins.add(p)

    pins_of_interest = pins_of_interest - non_present_pins

    logger.info("VDD nets: {}".format(", ".join([p for p, v in constant_input_pins.items() if v])))
    logger.info("GND nets: {}".format(", ".join([p for p, v in constant_input_pins.items() if not v])))

    if user_input_nets is None:
        user_input_nets = set()
    else:
        user_input_nets = set(user_input_nets)

    logger.info("User supplied input nets: {}".format(user_input_nets))
    deduced_input_nets = _find_input_gates(graph)
    logger.info("Additional detected input nets: {}".format(deduced_input_nets))
    input_nets = user_input_nets | deduced_input_nets | constant_input_pins.keys()
    output_nodes = pins_of_interest - input_nets
    logger.info("Detected input nets: {}".format(input_nets))
    logger.info("Detected output nets: {}".format(output_nodes))

    # Find conductivity conditions for all pull-up/pull-down paths.
    conductivity_conditions, inputs = complex_cmos_graph_to_formula(graph,
                                                                    input_pins=input_nets,
                                                                    output_nodes=output_nodes,
                                                                    )
    # print('conductivity_conditions = ', conductivity_conditions)

    # Pretty-print the conductivity conditions.
    print()
    print('Conductivity conditions')
    for pin_a, conditions in conductivity_conditions.items():
        print(' ', pin_a)
        for pin_b, condition in conditions.items():
            print('  |-', pin_b, 'when', condition)
    print()

    # Add known values for VDD, GND
    constants = {sympy.Symbol(k): v for k, v in constant_input_pins.items()}
    output_symbols = {sympy.Symbol(n) for n in output_nodes}

    # Convert keys into symbols.
    conductivity_conditions = {sympy.Symbol(k): v for k, v in conductivity_conditions.items()}

    # Derive logic formulas from conductivity condition.
    logger.debug("Derive logic formulas from conductivity condition.")
    formulas_high = dict()
    formulas_low = dict()
    for output, cc in conductivity_conditions.items():
        logger.debug(f"Derive logic formulas from conductivity condition for '{output}'.")
        or_terms_high = []
        or_terms_low = []
        for input_pin, condition in cc.items():
            assert isinstance(input_pin, sympy.Symbol)
            assert isinstance(condition, boolalg.Boolean)
            # Simplify by inserting constants.
            condition = simplify_logic(condition.subs(constants))
            input_pin = simplify_logic(input_pin.subs(constants))
            # if `condition` then output is `connected` to `input_pin`.
            connected_to_high = simplify_logic(condition & input_pin)  # Is there a path to HIGH?
            connected_to_low = simplify_logic(condition & ~input_pin)  # Is there a path to LOW?
            or_terms_high.append(connected_to_high)
            or_terms_low.append(connected_to_low)

        # Logic OR of all elements.
        formulas_high[output] = simplify_logic(sympy.Or(*or_terms_high))
        formulas_low[output] = simplify_logic(sympy.Or(*or_terms_low))

    # Simplify formulas by substituting VDD and GND with known values.
    logger.debug("Simplify all formulas.")
    formulas_high = {k: simplify_logic(f.subs(constants)) for k, f in formulas_high.items()}
    formulas_low = {k: simplify_logic(f.subs(constants)) for k, f in formulas_low.items()}

    logger.debug('formulas_high = {}'.format(formulas_high))
    logger.debug('formulas_low = {}'.format(formulas_low))

    # Convert from strings into sympy symbols.
    inputs = {sympy.Symbol(i) for i in inputs} - set(constants.keys())
    logger.debug('inputs = {}'.format(inputs))

    # # Simplify formulas by substitution.
    # formulas_high_simplified = {k: _resolve_intermediate_variables(formulas_high, inputs, f)
    #                             for k, f in formulas_high.items()}
    #
    # formulas_low_simplified = {k: _resolve_intermediate_variables(formulas_low, inputs, f)
    #                            for k, f in formulas_low.items()}
    #
    # print('simplified formulas_high = {}'.format(formulas_high_simplified))
    # print('simplified formulas_low = {}'.format(formulas_low_simplified))

    # Find formulas for nets that are complementary and can never be in a high-impedance nor short-circuit state.
    complementary_formulas = {k: formulas_high[k]
                              for k in formulas_high.keys()
                              if _is_complementary(formulas_high[k], formulas_low[k])}

    # Use the complementary nets to simplify the other formulas.
    formulas_high = {n: f.subs(complementary_formulas) for n, f in formulas_high.items()}
    formulas_low = {n: f.subs(complementary_formulas) for n, f in formulas_low.items()}

    # Find high-Z conditions.
    high_impedance_conditions = {k: simplify_logic(~formulas_high[k] & ~formulas_low[k])
                                 for k in formulas_high.keys()}

    print("Complementary nets:")
    for net, f in complementary_formulas.items():
        print(' ', net, ':', f)

    print("High impedance conditions:")
    for net, condition in high_impedance_conditions.items():
        print(' ', net, ':', condition)

    # Create dependency graph to detect feedback loops.
    dependency_graph_high = _formula_dependency_graph(formulas_high)
    dependency_graph_low = _formula_dependency_graph(formulas_low)
    # dependency_graph = nx.DiGraph(itertools.chain(dependency_graph_high.edges, dependency_graph_low.edges))
    dependency_graph = dependency_graph_high

    # import matplotlib.pyplot as plt
    # nx.draw_networkx(dependency_graph)
    # plt.draw()
    # plt.show()

    # Check for cycles.
    cycles = list(nx.simple_cycles(dependency_graph))
    logger.info("Number of feed-back loops: {}".format(len(cycles)))

    # # Print feedback loops.
    # for cycle in cycles:
    #     print()
    #     print("cycle: {}".format(cycle))
    #     for el in cycle:
    #         print(' --> {} = {}'.format(el, formulas_high[el]))
    #     print()

    # Collect all nets that belong to a memory cycle.
    nets_of_memory_cycles = {n for c in cycles for n in c}
    logger.debug(f"Nets of memory cycles: {nets_of_memory_cycles}")

    def derive_memory(memory_output_net: sympy.Symbol,
                      inputs: Set[sympy.Symbol],
                      formulas: Dict[sympy.Symbol, boolalg.Boolean]) -> Memory:
        """
        Detect a memory loop given the output net of the memory.
        :param memory_output_net:
        :param inputs: Input atoms (the input nets to the loop).
        :param formulas:
        :return:
        """
        logger.info("Derive memory from output net: {}".format(memory_output_net))
        memory_output_resolved = _resolve_intermediate_variables(formulas, inputs, memory_output_net)
        logger.debug("Memory output net {}  = {}".format(memory_output_net, memory_output_resolved))
        d, dp, dn = boolean_derivatives(memory_output_resolved, memory_output_net)
        write_condition = ~dp
        logger.debug("write_condition =", write_condition)
        oscillation_condition = dn

        debug = True
        if debug:
            # Find expressions for the memory output when the write condition is met.
            # Find all variable assignments such that the write condition is met.
            write_condition_models = list(satisfiable(write_condition, all_models=True))
            logger.debug("write_condition_models =", write_condition_models)
            if write_condition_models == [False]:
                logger.warning("Detected a memory loop that is not possible to write to.")

            # Find what data the memory will store for each write condition.
            mem_data = []
            for model in write_condition_models:
                # Now find the memory output once the write condition is met.
                data = simplify_logic(memory_output_resolved.subs(model))
                mem_data.append((model, data))

            # Pretty print write conditions and corresponding write data.
            if len(mem_data) > 0:
                print()
                print("Write data for all write conditions:")
                model, _ = mem_data[0]
                vars = list(model.keys())
                print("\t", "\t".join((str(v) for v in vars)), "\t:", f"data ({memory_output_net})")
                for model, data in mem_data:
                    print("\t", "\t".join((str(model[var]) for var in vars)), "\t:", data)
                print()

        write_data = simplify_with_assumption(write_condition, memory_output_resolved)
        logger.debug(f"write_data = {write_data}")

        if not sympy.satisfiable(~write_condition):
            logger.warning("This is not a true memory, it will never store anything.")

        return Memory(data=write_data, write_condition=write_condition, oscillation_condition=oscillation_condition)

    # Derive memory for all cycles with each possible node as memory output.
    memory_output_nets = set()
    memory_by_output_net = dict()  # Cache memory loops for later usage.
    for loop_id, cycle in enumerate(cycles):
        # print('cycle = {}'.format(cycle))
        nodes_in_cycle = set(cycle)
        _inputs = inputs | nets_of_memory_cycles - nodes_in_cycle
        for node in cycle:
            memory = derive_memory(node, _inputs, formulas_high)
            memory.loop_id = loop_id

            # print(memory)
            # print()

            if sympy.satisfiable(~memory.write_condition):
                # Store condition can be met -> this is a potential memory.
                memory_output_nets.add(node)
                memory_by_output_net[node] = memory

    logger.debug("Memory output nets: {}".format(memory_output_nets))

    # Solve equation system for output.
    # TODO: stop resolving at memory elements.
    def solve_output_nets() -> Tuple[Dict[boolalg.BooleanAtom, boolalg.Boolean], Dict[boolalg.BooleanAtom, Memory]]:
        output_formulas = dict()
        latches = dict()

        wavefront = {sympy.Symbol(o) for o in output_nodes}

        while wavefront:

            out = wavefront.pop()
            assert out not in output_formulas
            assert out not in latches

            new_wavefront = set()

            if out not in nets_of_memory_cycles:
                logger.debug(f"Find formula for {out}.")

                assert isinstance(out, boolalg.Boolean)
                formula = _resolve_intermediate_variables(formulas_high, inputs | nets_of_memory_cycles, out)
                output_formulas[out] = formula

                # Find new wavefront.
                atoms = formula.atoms(sympy.Symbol)
                new_wavefront.update(atoms)

                # Find inputs into formula that come from a memory.
                new_atoms = atoms - output_formulas.keys() - latches.keys() - inputs
                unknown_memory_nets = new_atoms & nets_of_memory_cycles
                logger.debug(f"Unknown inputs into {out} from memory: {unknown_memory_nets}")
            else:
                # Resolve memory.
                assert out in nets_of_memory_cycles

                # Output net of the memory loop.
                memory_net = out
                memory = memory_by_output_net[memory_net]

                latches[memory_net] = memory

                # Find new wavefront.
                atoms = memory.write_condition.atoms(sympy.Symbol) | memory.data.atoms(sympy.Symbol)
                new_wavefront.update(atoms)

            wavefront.update(new_wavefront - output_formulas.keys() - latches.keys() - inputs)

        return output_formulas, latches

    output_formulas, latches = solve_output_nets()

    # Simplify formulas in memories.
    for m in latches.values():
        m.data = simplify_logic(m.data.subs(output_formulas))
        m.write_condition = simplify_logic(m.write_condition.subs(output_formulas))
        m.oscillation_condition = simplify_logic(m.oscillation_condition.subs(output_formulas))

    logger.debug("Output functions ({}):".format(len(output_formulas)))
    output_combinatorial = dict()
    for net, formula in output_formulas.items():
        z = high_impedance_conditions[net]

        # Get the output function assuming that the output is not high-impedance.
        function = simplify_with_assumption(~z, formula)

        output_combinatorial[net] = CombinationalOutput(function=function, high_impedance=z)
        logger.debug(f"{net} = {function}, Z: {z}")

    logger.debug("Latches ({}):".format(len(latches)))

    for output_net, latch in latches.items():
        logger.info(f"{output_net} = {latch}")

    if not latches:
        logger.info("No latches found.")

    if len(cycles) > 0:
        assert len(latches) > 0, "Found feedback loops but no latches were derived."
    if len(latches):
        logger.info(f"Number of latches: {len(latches)}")

    return AbstractCircuit(output_symbols, output_combinatorial, latches)


class NetlistGen:
    """
    Generate test netlists.
    """

    def __init__(self):
        self.net_counter = itertools.count()
        self.vdd = "vdd"
        self.gnd = "gnd"

    def new_net(self, prefix=""):
        return "{}{}".format(prefix, next(self.net_counter))

    def inv(self, a, out):
        # Create transistors of a INV.
        return [
            (self.vdd, out, (a, ChannelType.PMOS)),
            (self.gnd, out, (a, ChannelType.NMOS)),
        ]

    def nor2(self, a, b, out):
        # Create transistors of a NOR.
        i = self.new_net("i")
        return [
            (self.vdd, i, (a, ChannelType.PMOS)),
            (i, out, (b, ChannelType.PMOS)),
            (out, self.gnd, (a, ChannelType.NMOS)),
            (out, self.gnd, (b, ChannelType.NMOS)),
        ]

    def nand2(self, a, b, out):
        # Create transistors of a NAND.
        i = self.new_net("i")
        return [
            (self.vdd, out, (a, ChannelType.PMOS)),
            (self.vdd, out, (b, ChannelType.PMOS)),
            (out, i, (a, ChannelType.NMOS)),
            (i, self.gnd, (b, ChannelType.NMOS)),
        ]

    def and2(self, a, b, out):
        nand_out = self.new_net(prefix="nand_out")
        return self.nand2(a, b, nand_out) + self.inv(nand_out, out)

    def or2(self, a, b, out):
        nor_out = self.new_net(prefix="nor_out")
        return self.nor2(a, b, nor_out) + self.inv(nor_out, out)

    def mux2(self, in0, in1, sel, out):
        sel0 = self.new_net(prefix='sel0')
        sel1 = sel
        in0_masked = self.new_net('in0_masked')
        in1_masked = self.new_net('in1_masked')

        return self.inv(sel1, sel0) + self.nand2(in0, sel0, in0_masked) + self.nand2(in1, sel1, in1_masked) + \
               self.nand2(in0_masked, in1_masked, out)

    def latch(self, clk, d_in, d_out):
        a_2_6 = self.new_net("a_2_6_")
        a_18_74 = self.new_net("a_18_74_")
        a_23_6 = self.new_net("a_23_6_")
        a_35_84 = self.new_net("a_35_84_")
        a_18_6 = self.new_net("a_18_6_")
        a_35_6 = self.new_net("a_35_6_")

        edges = [
            # Command for converting a raw SPICE transistor netlist into this format:
            # cat raw_netlist.sp | grep -v + | awk '{ print "(\"" $2 "\", \"" $4 "\", (\"" $3  "\", " $6 ")),"}' \
            # | sed 's/pmos/ChannelType.PMOS/g' | sed 's/nmos/ChannelType.NMOS/g'

            (self.vdd, a_2_6, (clk, ChannelType.PMOS)),
            (a_18_74, self.vdd, (d_in, ChannelType.PMOS)),
            (a_23_6, a_18_74, (a_2_6, ChannelType.PMOS)),
            (a_35_84, a_23_6, (clk, ChannelType.PMOS)),
            (self.vdd, a_35_84, (d_out, ChannelType.PMOS)),
            (self.gnd, a_2_6, (clk, ChannelType.NMOS)),
            (d_out, self.vdd, (a_23_6, ChannelType.PMOS)),
            (a_18_6, self.gnd, (d_in, ChannelType.NMOS)),
            (a_23_6, a_18_6, (clk, ChannelType.NMOS)),
            (a_35_6, a_23_6, (a_2_6, ChannelType.NMOS)),
            (self.gnd, a_35_6, (d_out, ChannelType.NMOS)),
            (d_out, self.gnd, (a_23_6, ChannelType.NMOS)),
        ]

        return edges


def test_analyze_circuit_graph():
    # Create CMOS network of a AND gate (NAND -> INV).

    gen = NetlistGen()
    edges = gen.and2('a', 'b', 'output')
    g = nx.MultiGraph()
    for e in edges:
        g.add_edge(*e)

    pins_of_interest = {'output'}
    known_pins = {'vdd': True, 'gnd': False}
    abstract = analyze_circuit_graph(g, pins_of_interest=pins_of_interest, constant_input_pins=known_pins)

    # Verify that the deduced boolean function is equal to the AND function.
    a, b = sympy.symbols('a, b')
    assert bool_equals(abstract.outputs[sympy.Symbol('output')].function, a & b)


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
    abstract = analyze_circuit_graph(g, pins_of_interest=pins_of_interest,
                                     constant_input_pins=known_pins,
                                     user_input_nets={'a', 'b'})

    # Verify that the deduced boolean function is equal to the XOR function.
    a, b = sympy.symbols('a, b')
    assert bool_equals(abstract.outputs[sympy.Symbol('c')].function, a ^ b)
    assert not abstract.latches


def test_analyze_circuit_graph_mux2():
    gen = NetlistGen()

    edges = []

    d0 = 'a'
    d1 = 'b'
    out = 'y'
    sel = 's'

    edges += gen.mux2(d0, d1, sel, out)

    g = nx.MultiGraph()
    for e in edges:
        g.add_edge(*e)

    pins_of_interest = {d0, d1, out, sel}
    known_pins = {'vdd': True, 'gnd': False}
    abstract = analyze_circuit_graph(g, pins_of_interest=pins_of_interest, constant_input_pins=known_pins)
    # Verify that the deduced boolean function is equal to the MUX function.

    a, b, s = sympy.symbols('a, b, s')
    assert bool_equals(abstract.outputs[sympy.Symbol('y')].function, (a & ~s) | (b & s))
    assert not abstract.latches


def test_analyze_circuit_graph_latch():
    g = nx.MultiGraph()
    # Network of a LATCH.
    gen = NetlistGen()
    edges = gen.latch('CLK', 'D', 'Q')
    for e in edges:
        g.add_edge(*e)

    pins_of_interest = {'CLK', 'D', 'Q'}
    known_pins = {'vdd': True, 'gnd': False}
    abstract = analyze_circuit_graph(g, pins_of_interest=pins_of_interest, constant_input_pins=known_pins)

    assert len(abstract.latches) == 1


def test_analyze_circuit_graph_set_reset_nand():
    # Network of a set-reset nand feedback loop.
    r"""
     S -----|\
            |&O--+--- Y1
        +---|/   |
        |        |
        | +------+
        | |
        +-|------+
          |      |
          +-|\   |
            |&O--+--- Y2
     R -----|/
    """

    gen = NetlistGen()

    g = nx.MultiGraph()
    edges = gen.nand2("S", "Y2", "Y1") + gen.nand2("R", "Y1", "Y2")
    for e in edges:
        g.add_edge(*e)

    pins_of_interest = {'S', 'R', 'Y1', 'Y2'}
    known_pins = {'vdd': True, 'gnd': False}
    abstract = analyze_circuit_graph(g, pins_of_interest=pins_of_interest, constant_input_pins=known_pins)
    print(abstract.outputs)
    print(abstract.latches)

    assert len(abstract.latches) == 1
    # TODO


def test_analyze_circuit_graph_dff_pos():
    gen = NetlistGen()

    edges = []

    d = 'D'
    q = 'Q'
    clk = 'CLK'
    clk_inv = gen.new_net(prefix='clk_inv')
    d_i = gen.new_net(prefix='d_i')

    edges += gen.inv(clk, clk_inv)
    edges += gen.latch(clk, d, d_i)
    edges += gen.latch(clk_inv, d_i, q)

    g = nx.MultiGraph()
    for e in edges:
        g.add_edge(*e)

    pins_of_interest = {clk, d, q}
    known_pins = {'vdd': True, 'gnd': False}
    abstract = analyze_circuit_graph(g, pins_of_interest=pins_of_interest, constant_input_pins=known_pins)

    assert len(abstract.latches) == 2


def test_analyze_circuit_graph_dff_pos_sync_reset():
    gen = NetlistGen()

    edges = []

    d = 'D'
    q = 'Q'
    r = 'R'
    clk = 'CLK'
    clk_inv = gen.new_net(prefix='clk_inv')
    d_i = gen.new_net(prefix='d_i')
    d_rst = gen.new_net(prefix='d_rst')

    edges += gen.and2(d, r, d_rst)
    edges += gen.inv(clk, clk_inv)
    edges += gen.latch(clk_inv, d_rst, d_i)
    edges += gen.latch(clk, d_i, q)

    g = nx.MultiGraph()
    for e in edges:
        g.add_edge(*e)

    pins_of_interest = {clk, d, q}
    known_pins = {'vdd': True, 'gnd': False}
    abstract = analyze_circuit_graph(g, pins_of_interest=pins_of_interest, constant_input_pins=known_pins)

    assert len(abstract.latches) == 2


def test_analyze_circuit_graph_dff_pos_scan():
    gen = NetlistGen()

    edges = []

    d = 'D'
    q = 'Q'
    clk = 'CLK'
    scan_enable = 'ScanEnable'
    scan_in = 'ScanIn'
    scan_mux_out = 'ScanMux_DO'
    clk_inv = gen.new_net(prefix='clk_inv')
    d_i = gen.new_net(prefix='d_i')

    edges += gen.inv(clk, clk_inv)

    edges += gen.mux2(d, scan_in, scan_enable, scan_mux_out)

    edges += gen.latch(clk, scan_mux_out, d_i)
    edges += gen.latch(clk_inv, d_i, q)

    g = nx.MultiGraph()
    for e in edges:
        g.add_edge(*e)

    pins_of_interest = {clk, d, q}
    known_pins = {'vdd': True, 'gnd': False}
    abstract = analyze_circuit_graph(g, pins_of_interest=pins_of_interest, constant_input_pins=known_pins)
    assert len(abstract.latches) == 2
