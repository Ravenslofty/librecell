##
## Copyright (c) 2019 Thomas Kramer.
## 
## This file is part of librecell-common 
## (see https://codeberg.org/tok/librecell/src/branch/master/librecell-common).
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
from . import spice_parser
from PySpice.Spice.Parser import SpiceParser

from PySpice.Spice.Parser import SubCircuitStatement

from copy import copy
from lclayout.data_types import *
import networkx as nx
from typing import Tuple, List, Set, Iterable


def get_subcircuit_ports(file: str, subckt_name: str) -> List[str]:
    """ Find port names of a subcircuit.
    :param file: Path to the spice file containing the subcircuit.
    :param subckt_name: Name of the subcircuit.
    :return: List of node names.
    """

    sc = load_subcircuit(file, subckt_name)
    return sc.nodes


def load_subcircuit(file: str, subckt_name: str) -> SubCircuitStatement:
    """ Load a sub circuit from a SPICE file.
    :param file: Path to the spice file containing the subcircuit.
    :param subckt_name: Name of the subcircuit.
    :return: The subcircuit.
    """

    parser = SpiceParser(path=file)
    name_match = [s for s in parser.subcircuits if s.name == subckt_name]

    if not name_match:
        raise Exception("No such sub circuit: {}".format(subckt_name))

    if len(name_match) == 1:
        return copy(name_match[0])

    raise Exception("Multiple definitions of sub circuit: {}".format(subckt_name))


def load_transistor_netlist(path: str, subckt_name: str) -> Tuple[List[Transistor], Set[str]]:
    """ Load a subcircuit from a spice netlist.

    Parameters
    ----------
    path: The path to the netlist.

    Returns
    -------
    Returns a list of `Transistor`s and a list of the pin names including power pins.
    (List[Transistors], pin_names)
    """

    with open(path) as f:
        source = f.read()

        ast = spice_parser.parse_spice(source)

        def get_channel_type(s):
            """Determine the channel type of transistor from the model name.
            """
            if s.lower().startswith('n'):
                return ChannelType.NMOS
            return ChannelType.PMOS

        match = [s for s in ast if s.name == subckt_name]

        if len(match) < 1:
            raise Exception("No valid subcircuit found in file with name '%s'." % subckt_name)

        circuit = match[0]

        # Get transistors
        transistors = [
            Transistor(get_channel_type(t.model_name), t.ns, t.ng, t.nd, channel_width=t.params['W'], name=t.name)
            for t in circuit.content if type(t) is spice_parser.MOSFET
        ]

        return transistors, circuit.ports


def is_ground_net(net: str) -> bool:
    """ Test if net is something like 'gnd' or 'vss'.
    """
    ground_nets = {0, '0', 'gnd', 'vss'}
    return net.lower() in ground_nets


def is_supply_net(net: str) -> bool:
    """ Test if net is something like 'vcc' or 'vdd'.
    """
    supply_nets = {'vcc', 'vdd'}
    return net.lower() in supply_nets


def is_power_net(net: str) -> bool:
    return is_ground_net(net) or is_supply_net(net)


def get_io_pins(pin_names: Iterable[str]) -> Set[str]:
    """ Get all pin names that don't look like power pins.
    """
    return {p for p in pin_names if not is_ground_net(p) and not is_supply_net(p)}


def get_cell_inputs(transistors: Iterable[Transistor]) -> Set[str]:
    """Given the transistors of a cell find the nets connected only to transistor gates.
    Will not work for transmission gates.
    """

    transistors = [t for t in transistors if t is not None]

    gate_nets = set(t.gate for t in transistors)
    source_and_drain_nets = set(t.left for t in transistors) | set(t.right for t in transistors)

    # Input nets are only connected to transistor gates.
    input_nets = gate_nets - source_and_drain_nets

    return input_nets


def _transistors2graph(transistors: Iterable[Transistor]) -> nx.MultiGraph:
    """ Create a graph representing the transistor network.
        Each edge corresponds to a transistor, each node to an electrical potential.
    """
    G = nx.MultiGraph()
    for t in transistors:
        G.add_edge(t.left, t.right, t)
    assert nx.is_connected(G)
    return G


def _is_output_net(net_name, power_nets: Iterable, transistor_graph: nx.MultiGraph) -> bool:
    """
    Determine if the net is a driven output net which is the case if there is a path from the net
    to a power rail.
    :param net_name: The net to be checked.
    :param power_nets: List of available power nets ["vdd", "gnd", ...].
    :param transistor_graph:
    :return: True, iff `net_name` is a OUTPUT net. False, iff it is a INOUT net.
    """

    return any((
        nx.has_path(transistor_graph, net_name, pn)
        for pn in power_nets
    ))
