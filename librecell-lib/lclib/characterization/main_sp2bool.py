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

from liberty import parser as liberty_parser
from liberty.types import *

from ..liberty import util as liberty_util
from ..logic import functional_abstraction
from ..logic import seq_recognition
from . import util

import argparse
from copy import deepcopy
from PySpice.Unit import *
from lccommon import net_util
from lccommon.net_util import load_transistor_netlist, is_ground_net, is_supply_net
import networkx as nx
import sympy
import re


def main():
    """
    Command-line tool for cell characterization.
    Currently only combinatorial cells are supported excluding tri-state cells.
    :return:
    """

    logger = logging.getLogger(__name__)
    logger.info("sp2bool main function")

    parser = argparse.ArgumentParser(
        description='Find boolean formulas that describe the behaviour of a CMOS circuit.',
        epilog='Example: sp2bool --spice INVX1.sp --cell INVX1')

    parser.add_argument('--cell', required=True, metavar='CELL_NAME', type=str, help='Cell name.')

    parser.add_argument('--spice', required=True, metavar='SPICE', type=str,
                        help='SPICE netlist containing a subcircuit with the same name as the cell.')

    parser.add_argument('--diff', required=False,
                        nargs="+",
                        metavar='DIFFERENTIAL_PATTERN',
                        type=str,
                        help='Specify differential inputs as "NonInverting,Inverting" tuples.'
                             'The placeholder "%" can be used like "%_P,%_N" or "%,%_Diff", ...')

    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--plot-network', action='store_true',
                        help='Show a plot of the transistor graph for debugging. '
                             'Transistors are edges in the graph. The edges are labelled with the gate net.')

    # Parse arguments
    args = parser.parse_args()

    DEBUG = args.debug
    log_level = logging.DEBUG if DEBUG else logging.INFO

    if DEBUG:
        log_format = '%(module)16s %(levelname)8s: %(message)s'
    else:
        # Also output name of function in DEBUG mode.
        # log_format = '%(module)16s %(funcName)16s %(levelname)8s: %(message)s'
        log_format = '%(message)s'

    logging.basicConfig(format=log_format, level=log_level)

    cell_name = args.cell

    # Load netlist of cell
    netlist_path = args.spice
    logger.info('Load netlist: %s', netlist_path)
    transistors_abstract, cell_pins = load_transistor_netlist(netlist_path, cell_name)
    io_pins = net_util.get_io_pins(cell_pins)

    logger.info(f"Cell pins: {cell_pins}")
    print(transistors_abstract)

    # Detect power pins.
    # TODO: don't decide based only on net name.
    power_pins = [p for p in cell_pins if net_util.is_power_net(p)]
    assert len(power_pins) == 2, "Expected to have 2 power pins."
    vdd_pins = [p for p in power_pins if net_util.is_supply_net(p)]
    gnd_pins = [p for p in power_pins if net_util.is_ground_net(p)]
    assert len(vdd_pins) == 1, "Expected to find one VDD pin but found: {}".format(vdd_pins)
    assert len(gnd_pins) == 1, "Expected to find one GND pin but found: {}".format(gnd_pins)
    vdd_pin = vdd_pins[0]
    gnd_pin = gnd_pins[0]

    logger.info(f"Supply net: {vdd_pin}")
    logger.info(f"Ground net: {gnd_pin}")

    # Match differential inputs.
    if args.diff is not None:
        differential_inputs = util.find_differential_inputs_by_pattern(args.diff, io_pins)
    else:
        differential_inputs = dict()

    # Sanity check.
    if len(set(differential_inputs.keys())) != len(set(differential_inputs.values())):
        logger.error(f"Mismatch in the mapping of differential inputs.")
        exit(1)

    def _transistors2multigraph(transistors) -> nx.MultiGraph:
        """ Create a graph representing the transistor network.
            Each edge corresponds to a transistor, each node to a net.
        """
        G = nx.MultiGraph()
        for t in transistors:
            G.add_edge(t.source_net, t.drain_net, (t.gate_net, t.channel_type))

        connected = list(nx.connected_components(G))
        print(connected)
        logger.debug(f"Number of connected graphs: {len(connected)}")
        assert nx.is_connected(G)
        return G

    # Derive boolean functions for the outputs from the netlist.
    logger.info("Derive boolean functions for the outputs based on the netlist.")
    transistor_graph = _transistors2multigraph(transistors_abstract)
    if args.plot_network:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(transistor_graph)
        nx.draw(transistor_graph, pos, with_labels=True)
        print(list(transistor_graph.edges(keys=True)))
        edge_labels = {(a, b): gate_net for a, b, (gate_net, channel_type) in transistor_graph.edges(keys=True)}
        nx.draw_networkx_edge_labels(transistor_graph, pos, edge_labels=edge_labels)
        plt.show()
    abstract = functional_abstraction.analyze_circuit_graph(graph=transistor_graph,
                                                            pins_of_interest=io_pins,
                                                            constant_input_pins={vdd_pin: True,
                                                                                 gnd_pin: False},
                                                            differential_inputs=differential_inputs,
                                                            user_input_nets=None)
    output_functions_deduced = abstract.outputs

    # Convert keys into strings (they are `sympy.Symbol`s now)
    output_functions_deduced = {output.name: comb.function for output, comb in output_functions_deduced.items()}

    # Log deduced output functions.
    for output_name, function in output_functions_deduced.items():
        logger.info("Deduced output function: {} = {}".format(output_name, function))

    # Try to recognize sequential cells.
    seq = seq_recognition.extract_sequential_circuit(abstract)

    if seq is not None:
        print()
        print(seq)
