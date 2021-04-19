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

from lclayout.data_types import *
import networkx as nx
from typing import Tuple, List, Set, Iterable
import klayout.db as db

import logging

logger = logging.getLogger(__name__)


def get_subcircuit_ports(file: str, subckt_name: str) -> List[str]:
    """ Find port names of a subcircuit.
    :param file: Path to the spice file containing the subcircuit.
    :param subckt_name: Name of the subcircuit.
    :return: List of node names.
    """

    _, sc = load_subcircuit(file, subckt_name)

    pins = [p.name() for p in sc.each_pin()]

    return pins


def load_subcircuit(path: str, circuit_name: str) -> Tuple[db.Netlist, db.Circuit]:
    """ Load a sub circuit from a SPICE file.
    :param path: Path to the spice file containing the subcircuit.
    :param circuit_name: Name of the subcircuit.
    :return: A tuple with the netlist and the circuit. Returns none if there's no subcircuit with this name.
    """

    netlist = db.Netlist()
    netlist.read(path, db.NetlistSpiceReader())
    if not circuit_name.isupper():
        # KLayout converts cell names to uppercase.
        # Check if that is still true:
        assert netlist.circuit_by_name(circuit_name) is None, "KLayout did not convert cell names to upper case."
        logger.info(f"Convert non-upper case cellname to upper case: '{circuit_name}' -> '{circuit_name.upper()}'")
        circuit_name = circuit_name.upper()

    circuit: db.Circuit = netlist.circuit_by_name(circuit_name)

    # Have to return the netlist too. Otherwise it is deconstructed already.
    return netlist, circuit


def load_transistor_netlist(path: str, circuit_name: str) -> Tuple[List[Transistor], Set[str]]:
    """ Load a transistor level circuit from a spice netlist.

    :param path: The path to the netlist.

    Returns
    -------
    Returns a list of `Transistor`s and a list of the pin names including power pins.
    (List[Transistors], pin_names)
    """

    # Read netlist. TODO: take netlist object as argument.
    netlist, circuit = load_subcircuit(path, circuit_name)

    if circuit is None:
        all_circuits = [c.name for c in netlist.each_circuit()]
        logger.error("No such circuit: {}".format(circuit_name))
        logger.info(f"Circuits in netlist: {all_circuits}")
        raise Exception("No such circuit: {}".format(circuit_name))

    pins = [p.name() for p in circuit.each_pin()]

    def get_channel_type(s: str):
        """Determine the channel type of transistor from the model name.
        """
        if s.lower().startswith('n'):
            return ChannelType.NMOS
        return ChannelType.PMOS

    mos4 = db.DeviceClassMOS4Transistor()
    id_gate_4 = mos4.terminal_id('G')
    id_source_4 = mos4.terminal_id('S')
    id_drain_4 = mos4.terminal_id('D')

    transistors_klayout = [
        Transistor(get_channel_type(d.device_class().name),
                   d.net_for_terminal(id_source_4).name,
                   d.net_for_terminal(id_gate_4).name,
                   d.net_for_terminal(id_drain_4).name,
                   channel_width=d.parameter('W') * 1e-6,  # Convert into micrometers.
                   name=d.name
                   )
        for d in circuit.each_device()
        if isinstance(d.device_class(), db.DeviceClassMOS3Transistor)
           or isinstance(d.device_class(), db.DeviceClassMOS4Transistor)]

    # with open(path) as f:
    #     source = f.read()
    #
    #     ast = spice_parser.parse_spice(source)
    #
    #     match = [s for s in ast if s.name == subckt_name]
    #
    #     if len(match) < 1:
    #         raise Exception("No valid subcircuit found in file with name '%s'." % subckt_name)
    #
    #     circuit = match[0]
    #
    #     # Get transistors
    #     transistors = [
    #         Transistor(get_channel_type(t.model_name), t.ns, t.ng, t.nd, channel_width=t.params['W'], name=t.name)
    #         for t in circuit.content if type(t) is spice_parser.MOSFET
    #     ]
    #
    #     for t in transistors_klayout:
    #         print(t.channel_width)
    #
    #     for t in transistors:
    #         print(t.channel_width)
    #
    #     return transistors, circuit.ports

    return transistors_klayout, set(pins)


def is_ground_net(net: str) -> bool:
    """ Test if net is something like 'gnd' or 'vss'.
    """
    ground_nets = {0, '0', 'gnd', 'vss', 'vgnd'}
    return net.lower() in ground_nets


def is_supply_net(net: str) -> bool:
    """ Test if net is something like 'vcc' or 'vdd'.
    """
    supply_nets = {'vcc', 'vdd', 'vpwr'}
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

    gate_nets = set(t.gate_net for t in transistors)
    source_and_drain_nets = set(t.source_net for t in transistors) | set(t.drain_net for t in transistors)

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
