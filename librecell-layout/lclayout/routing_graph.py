##
## Copyright (c) 2019 Thomas Kramer.
## 
## This file is part of librecell-layout 
## (see https://codeberg.org/tok/librecell/src/branch/master/librecell-layout).
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the CERN Open Hardware License (CERN OHL-S) as it will be published
## by the CERN, either version 2.0 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## CERN Open Hardware License for more details.
## 
## You should have received a copy of the CERN Open Hardware License
## along with this program. If not, see <http://ohwr.org/licenses/>.
## 
## 
##
import networkx as nx
from itertools import count

from .layout.grid_helpers import *
from .layout.geometry_helpers import *
from .layout.grid import Grid2D
from .layout.transistor import TransistorLayout
from .data_types import Transistor
from . import tech_util

from typing import Any, Dict, List, Tuple, Iterable, Optional
import logging

logger = logging.getLogger(__name__)


def create_routing_graph_base(grid: Grid2D, tech) -> nx.Graph:
    """ Construct the full mesh of the routing graph.
    :param grid_points: set of grid points
    :param tech: module containing technology information
    :return: nx.Graph
    """
    logging.debug('Create routing graph.')

    # Create routing graph.

    # Create nodes and vias.
    G = nx.Graph()

    # Create nodes on routing layers.
    for layer, directions in tech.routing_layers.items():
        for p in grid:
            n = layer, p
            G.add_node(n)

    # Create via edges.
    for (l1, l2), via_layer in tech.via_layers.items():
        if l1 in tech.routing_layers and l2 in tech.routing_layers:
            for p in grid:
                n1 = (l1, p)
                n2 = (l2, p)

                G.add_edge(n1, n2,
                           weight=tech.via_weights[(l1, l2)],
                           multi_via=tech.multi_via.get((l1, l2), 1)
                           )

    # Create intra layer routing edges.
    for layer, directions in tech.routing_layers.items():
        for p1 in grid:
            x1, y1 = p1
            x2 = x1 + tech.routing_grid_pitch_x
            y2 = y1 + tech.routing_grid_pitch_y

            # ID of the graph node.
            n = layer, p1

            # Horizontal edge.
            if 'h' in directions:
                n_right = layer, (x2, y1)
                if n_right in G.nodes:
                    weight = tech.weights_horizontal[layer] * abs(x2 - x1)
                    G.add_edge(n, n_right, weight=weight, orientation='h')

            # Vertical edge.
            if 'v' in directions:
                n_upper = layer, (x1, y2)
                if n_upper in G.nodes:
                    weight = tech.weights_vertical[layer] * abs(y2 - y1)
                    G.add_edge(n, n_upper, weight=weight, orientation='v')

    return G


def prepare_routing_nodes(G: nx.Graph, grid: Grid2D, shapes: Dict[Any, pya.Region], tech):
    """ Get legal routing nodes for each routing layer by removing nodes that would conflict
    with predefined `shapes`.
    :param grid: The routing grid (Grid2D).
    :param shapes: Dict[layer name, pya.Shapes]
    :param tech: module containing technology information
    :return: Dict[layer name, List[Node]]
    """

    # Build a spacing rule graph by mapping the minimal spacing between layer a and layer b to an edge
    # a-b in the graph with weight=min_spacing.
    spacing_graph = tech_util.spacing_graph(tech.min_spacing)

    routing_nodes = dict()
    # Create routing grid and remove nodes that interact with some layers.
    for l in tech.routing_layers.keys():

        # Find nodes that interact with a blocking layer.
        illegal_nodes = set()

        if l in spacing_graph:
            other_layers = spacing_graph[l]

            w1 = (tech.wire_width.get(l, 1) + 1) // 2
            for other_layer in other_layers:
                if other_layer not in spacing_graph:
                    # No spacing defined for this layer.
                    continue

                if l == other_layer:
                    # Intra layer spacings are not handled here.
                    continue

                w2 = (tech.wire_width.get(other_layer, 0) + 1) // 2
                spacing = spacing_graph[l][other_layer]['min_spacing']
                margin = w1 + w2 + spacing
                r_other = pya.Region(shapes[other_layer])
                illegal = interacting(grid, r_other, margin)
                illegal_nodes.update(illegal)

            G.remove_nodes_from(((l, p) for p in illegal_nodes))

        routing_nodes[l] = set(grid) - illegal_nodes

    return routing_nodes


def remove_existing_routing_edges(G: nx.Graph, shapes: Dict[Any, pya.Region], tech) -> None:
    """ Remove edges in G that are already routed by a shape in `shapes`.
    :param G: Routing graph to be modified.
    :param shapes: Dict[layer, pya.Shapes]
    :param tech: module containing technology information
    :return: None
    """

    # Remove all routing edges that are inside existing shapes.
    # (They are already connected and cannot be used for routing).
    for l in tech.routing_layers.keys():
        edges = edges_inside(G, pya.Region(shapes[l]), 1)
        for e in edges:
            (l1, _), (l2, _) = e
            if (l1, l2) == (l, l):
                G.remove_edge(*e)


def extract_terminal_nodes(routing_nodes: List[Tuple[str, str, Tuple[int, int]]],
                           net_regions: Dict[str, List[pya.Region]],
                           tech):
    """ Get terminal nodes for each net.
    :param routing_nodes: Legal routing nodes for each layer.
    :param net_regions: Regions that are connected to a net: Dict[net, Dict[layer, pya.Region]]
    :param tech: module containing technology information
    :return: list of terminals: [(net, layer, [terminal, ...]), ...]
    """

    # Create a list of terminal areas: [(net, layer, [terminal, ...]), ...]
    terminals_by_net = []
    for net, regions in net_regions.items():
        for layer, region in regions.items():
            for net_shape in region.each_merged():

                possible_via_layers = [v for l, v in tech.via_layers.items() if layer in l]
                enc = max((tech.minimum_enclosure.get((layer, via_layer), 0) for via_layer in possible_via_layers))
                max_via_size = max((tech.via_size[l] for l in possible_via_layers))

                if layer in tech.routing_layers:
                    # On routing layers enclosure can be added, so nodes are not required to be properly enclosed.
                    d = 1
                else:
                    # A routing node must be properly enclosed to be used.
                    d = enc + max_via_size // 2

                routing_terminals = inside(routing_nodes[layer], pya.Region(net_shape), d)
                terminals_by_net.append((net, layer, routing_terminals))
                # Don't use terminals for normal routing
                routing_nodes[layer] -= set(routing_terminals)
                # TODO: need to be removed from G also. Better: construct edges in G afterwards.
                # G.remove_nodes_from((('pc',p) for p in routing_terminals))

    # Sanity check
    error = False
    for net_name, layer, terminals in terminals_by_net:
        if len(terminals) == 0:
            logger.error("Shape of net {} does not contain any routing grid point.".format(net_name))
            error = True

    return terminals_by_net


def embed_transistor_terminal_nodes(G: nx.Graph,
                                    terminals_by_net: List[Tuple[str, str, List[Tuple[int, int]]]],
                                    transistor_layouts: Dict[Transistor, TransistorLayout],
                                    tech):
    """ Embed the terminal nodes of a the transistors into the routing graph.
    Modifies `G` and `terminals_by_net`
    :param G: The routing graph.
    :param terminals_by_net:
    :param transistor_layouts: List[TransistorLayout]
    :param tech: module containing technology information
    :return: None
    """
    # Connect terminal nodes of transistor gates in G.
    for t_layout in transistor_layouts.values():
        terminals = t_layout.terminals
        for net, ts in terminals.items():
            for t in ts:
                layer, (x, y) = t

                # Insert terminal into G.
                next_x = grid_round(x, tech.grid_offset_x, tech.routing_grid_pitch_x)

                assert next_x == x, Exception("Terminal node not x-aligned.")

                x_aligned_nodes = [(l, (_x, y)) for l, (_x, y) in G if l == layer and _x == x]

                def dist(a, b):
                    _, (x1, y1) = a
                    _, (x2, y2) = b
                    return (x1 - x2) ** 2 + (y1 - y2) ** 2

                neighbour_node = min(x_aligned_nodes, key=lambda n: dist(n, t))

                # TODO: weight proportional to gate width?
                G.add_edge(t, neighbour_node, weight=1000, wire_width=tech.gate_length)

            coords = [c for _, c in ts]
            terminals_by_net.append((net, layer, coords))


def create_virtual_terminal_nodes(G: nx.Graph,
                                  routing_nodes,
                                  terminals_by_net: List[Tuple[str, str, Tuple[int, int]]],
                                  io_pins: Iterable,
                                  tech):
    """ Create virtual terminal nodes for each net.
    :param G: The routing graph. Will be modified.
    :param routing_nodes:
    :param terminals_by_net:
    :param io_pins: Names of the I/O nets.
    :param tech: module containing technology information
    :return: Returns a set of virtual terminal nodes: Dict[('virtual...', net, layer, id)]
    """
    # Create virtual graph nodes for each net terminal.
    virtual_terminal_nodes = {}
    cnt = count()

    for net, layer, terminals in terminals_by_net:
        weight = 1000
        if len(terminals) > 0:
            if layer == 'l_active' and False:  # TODO: make tech independet
                for p in terminals:
                    # Force router to connect to all contacts to a l_active shape.
                    virtual_net_terminal = ('virtual', net, layer, next(cnt))
                    virtual_terminal_nodes.setdefault(net, []).append(virtual_net_terminal)
                    n = layer, p
                    assert n in G.nodes, "Node not present in graph: %s" % str(n)
                    G.add_edge(virtual_net_terminal, n, weight=weight)
            else:
                virtual_net_terminal = ('virtual', net, layer, next(cnt))
                virtual_terminal_nodes.setdefault(net, []).append(virtual_net_terminal)

                for p in terminals:
                    n = layer, p
                    assert n in G.nodes, "Node not present in graph: %s" % str(n)
                    # High weight for virtual edge
                    # TODO: High weight only for low-resistance layers.
                    G.add_edge(virtual_net_terminal, n, weight=weight)

    cnt = count()
    # Create virtual nodes for I/O pins.
    for p in io_pins:
        virtual_net_terminal = ('virtual_pin', p, tech.pin_layer, next(cnt))
        virtual_terminal_nodes.setdefault(p, []).append(virtual_net_terminal)

        for p in routing_nodes[tech.pin_layer]:
            n = tech.pin_layer, p
            x, y = p
            assert n in G.nodes, "Node not present in graph: %s" % str(n)
            # A huge weight assures that the virtual node is not used as a worm hole for routing.
            weight = 100000 + (y - tech.unit_cell_height // 2) // tech.routing_grid_pitch_y
            G.add_edge(virtual_net_terminal, n, weight=weight)

    return virtual_terminal_nodes
