#
# Copyright 2019-2020 Thomas Kramer.
#
# This source describes Open Hardware and is licensed under the CERN-OHL-S v2.
#
# You may redistribute and modify this documentation and make products using it
# under the terms of the CERN-OHL-S v2 (https:/cern.ch/cern-ohl).
# This documentation is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY,
# INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
# Please see the CERN-OHL-S v2 for applicable conditions.
#
# Source location: https://codeberg.org/tok/librecell
#
import networkx as nx
from itertools import count
from collections import defaultdict

from .layout.grid_helpers import *
from .layout.geometry_helpers import *
from .layout.grid import Grid2D
from .layout.layers import *
from .layout.transistor import TransistorLayout
from .data_types import Transistor
from . import tech_util

from typing import Any, Dict, List, Tuple, Iterable, Set
import logging

logger = logging.getLogger(__name__)


def create_routing_graph_base(grid: Grid2D, tech) -> nx.Graph:
    """ Construct the full mesh of the routing graph.
    :param grid: grid points
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
    for l1, l2, data in via_layers.edges(data=True):
        via_layer = data['layer']
        for p in grid:
            n1 = (l1, p)
            n2 = (l2, p)

            weight = tech.via_weights.get((l1, l2))
            if weight is None:
                weight = tech.via_weights[(l2, l1)]

            multi_via = tech.multi_via.get((l1, l2))
            if multi_via is None:
                multi_via = tech.multi_via.get((l2, l1), 1)

            # Create edge: n1 -- n2
            G.add_edge(n1, n2,
                       weight=weight,
                       multi_via=multi_via,
                       layer=via_layer
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
                    G.add_edge(n, n_right, weight=weight, orientation='h', layer=layer)

            # Vertical edge.
            if 'v' in directions:
                n_upper = layer, (x1, y2)
                if n_upper in G.nodes:
                    weight = tech.weights_vertical[layer] * abs(y2 - y1)
                    G.add_edge(n, n_upper, weight=weight, orientation='v', layer=layer)

    assert nx.is_connected(G)
    return G


def _get_routing_node_locations_per_layer(g: nx.Graph) -> Dict[Any, Set[Tuple[int, int]]]:
    """ For each layer extract the positions of the routing nodes.

    :param g: Routing graph.
    :return: Dict[layer name, set of (x,y) coordinates of routing nodes]
    """
    # Dict that will contain for each layer the node coordinates that can be used for routing.
    routing_nodes = defaultdict(set)
    # Populate `routing_nodes`
    for e in g.edges:
        (l1, p1), (l2, p2) = e
        routing_nodes[l1].add(p1)
        routing_nodes[l2].add(p2)

    return routing_nodes


def remove_illegal_routing_edges(graph: nx.Graph, shapes: Dict[Any, pya.Shapes], tech) -> None:
    """ Remove nodes and edges from  G that would conflict
    with predefined `shapes`.
    :param graph: routing graph.
    :param shapes: Dict[layer name, pya.Shapes]
    :param tech: module containing technology information
    :return: Dict[layer name, List[Node]]
    """

    # Build a spacing rule graph by mapping the minimal spacing between layer a and layer b to an edge
    # a-b in the graph with weight=min_spacing.
    spacing_graph = tech_util.spacing_graph(tech.min_spacing)

    # Get a dict mapping layer names to pya.Regions
    regions = {l: pya.Region(s) for l, s in shapes.items()}
    illegal_edges = set()
    # For each edge in the graph check if it conflicts with an existing shape.
    # Remember the edge if it is in conflict.
    for e in graph.edges:
        (l1, p1), (l2, p2) = e
        is_via = l1 != l2

        if not is_via:
            layer = l1
            other_layers = spacing_graph[layer]
            for other_layer in other_layers:
                if other_layer != layer:
                    min_spacing = spacing_graph[layer][other_layer]['min_spacing']
                    wire_width_half = (tech.wire_width.get(layer, 0) + 1) // 2
                    margin = wire_width_half + min_spacing
                    # TODO: treat horizontal and vertical lines differently if they don't have the same wire width.
                    region = regions[other_layer]
                    is_illegal_edge = interacts(p1, region, margin) or interacts(p2, region, margin)

                    if is_illegal_edge:
                        illegal_edges.add(e)
        else:
            assert p1 == p2, "End point coordinates of a via edge must match."
            layer = via_layers[l1][l2]['layer']
            if layer in spacing_graph:
                other_layers = spacing_graph[layer]
                for other_layer in other_layers:
                    if other_layer != layer:
                        if layer in spacing_graph and other_layer in spacing_graph:
                            min_spacing = spacing_graph[layer][other_layer]['min_spacing']
                            via_width_half = (tech.via_size[layer] + 1) // 2
                            margin = via_width_half + min_spacing
                            region = regions[other_layer]
                            is_illegal_edge = interacts(p1, region, margin)

                            if is_illegal_edge:
                                illegal_edges.add(e)

    # Now remove all edges from G that are in conflict with existing shapes.
    graph.remove_edges_from(illegal_edges)

    # Remove unconnected nodes.
    unconnected = set()
    for n in graph:
        d = nx.degree(graph, n)
        if d < 1:
            unconnected.add(n)
    graph.remove_nodes_from(unconnected)


def remove_existing_routing_edges(G: nx.Graph, shapes: Dict[Any, pya.Shapes], tech) -> None:
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


def extract_terminal_nodes(graph: nx.Graph,
                           shapes: Dict[str, pya.Shapes],
                           tech):
    """ Get terminal nodes for each net.
    :param graph: Routing graph.
    :param net_regions: Regions that are connected to a net: Dict[net, Dict[layer, pya.Region]]
    :param tech: module containing technology information
    :return: list of terminals: [(net, layer, [terminal, ...]), ...]
    """

    routing_nodes = _get_routing_node_locations_per_layer(graph)

    # Create a list of terminal areas: [(net, layer, [terminal, ...]), ...]
    terminals_by_net = []
    for layer, _shapes in shapes.items():
        for net_shape in _shapes.each():
            net = net_shape.property('net')

            if net is not None:
                possible_via_layers = [data['layer'] for _, _, data in via_layers.edges(layer, data=True)]
                enc = max((tech.minimum_enclosure.get((layer, via_layer), 0) for via_layer in possible_via_layers))
                max_via_size = max((tech.via_size[l] for l in possible_via_layers))

                # TODO: How to convert db.Shape into db.Region in a clean way???

                s = db.Shapes()
                s.insert(net_shape)
                terminal_region = pya.Region(s)

                if layer in tech.routing_layers:
                    # On routing layers enclosure can be added, so nodes are not required to be properly enclosed.
                    d = 1
                    routing_terminals = interacting(routing_nodes[layer], terminal_region, d)
                else:
                    # A routing node must be properly enclosed to be used.
                    d = enc + max_via_size // 2
                    routing_terminals = inside(routing_nodes[layer], terminal_region, d)

                terminals_by_net.append((net, layer, routing_terminals))
                # Don't use terminals for normal routing
                routing_nodes[layer] -= set(routing_terminals)
                # TODO: need to be removed from G also. Better: construct edges in G afterwards.

    # for net, regions in net_regions.items():
    #     for layer, region in regions.items():
    #         if layer in routing_nodes:
    #             for net_shape in region.each_merged():
    #
    #                 possible_via_layers = [data['layer'] for _, _, data in via_layers.edges(layer, data=True)]
    #                 enc = max((tech.minimum_enclosure.get((layer, via_layer), 0) for via_layer in possible_via_layers))
    #                 max_via_size = max((tech.via_size[l] for l in possible_via_layers))
    #
    #                 if layer in tech.routing_layers:
    #                     # On routing layers enclosure can be added, so nodes are not required to be properly enclosed.
    #                     d = 1
    #                     routing_terminals = interacting(routing_nodes[layer], pya.Region(net_shape), d)
    #                 else:
    #                     # A routing node must be properly enclosed to be used.
    #                     d = enc + max_via_size // 2
    #                     routing_terminals = inside(routing_nodes[layer], pya.Region(net_shape), d)
    #
    #                 terminals_by_net.append((net, layer, routing_terminals))
    #                 # Don't use terminals for normal routing
    #                 routing_nodes[layer] -= set(routing_terminals)
    #                 # TODO: need to be removed from G also. Better: construct edges in G afterwards.
    #         else:
    #             logger.warning("Layer '{}' does not contain any routing nodes.".format(layer))

    # # Sanity check
    # error = False
    # for net_name, layer, terminals in terminals_by_net:
    #     if len(terminals) == 0:
    #         logger.error(
    #             "Shape of net {} on layer '{}' does not contain any routing grid point.".format(net_name, layer))
    #         error = True

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
        terminals = t_layout.terminal_nodes()
        for net, ts in terminals.items():
            for t in ts:
                layer, (x, y) = t

                logger.info(f"Terminal node {net} {layer} {t}")

                # Insert terminal into G.
                next_x = grid_round(x, tech.routing_grid_pitch_x, tech.grid_offset_x)

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
                                  terminals_by_net: List[Tuple[str, str, Tuple[int, int]]],
                                  io_pins: Iterable,
                                  tech):
    """ Create virtual terminal nodes for each net.
    :param G: The routing graph. Will be modified.
    :param terminals_by_net:
    :param io_pins: Names of the I/O nets.
    :param tech: module containing technology information
    :return: Returns a set of virtual terminal nodes: Dict[('virtual...', net, layer, id)]
    """

    # Extract all routing nodes for each layer.
    routing_nodes = _get_routing_node_locations_per_layer(G)

    # Create virtual graph nodes for each net terminal.
    virtual_terminal_nodes = defaultdict(list)
    cnt = count()

    for net, layer, terminals in terminals_by_net:
        weight = 1000
        if len(terminals) > 0:
            # if layer in (l_ndiffusion, l_pdiffusion) and False:  # TODO: make tech independet
            #     for p in terminals:
            #         # Force router to connect to all contacts to a l_active shape.
            #         virtual_net_terminal = ('virtual', net, layer, next(cnt))
            #         virtual_terminal_nodes[net].append(virtual_net_terminal)
            #         n = layer, p
            #         assert n in G.nodes, "Node not present in graph: %s" % str(n)
            #         G.add_edge(virtual_net_terminal, n, weight=weight)
            # else:
            virtual_net_terminal = ('virtual', net, layer, next(cnt))
            virtual_terminal_nodes[net].append(virtual_net_terminal)

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
        virtual_terminal_nodes[p].append(virtual_net_terminal)

        for p in routing_nodes[tech.pin_layer]:
            n = tech.pin_layer, p
            x, y = p
            assert n in G.nodes, "Node not present in graph: %s" % str(n)
            # A huge weight assures that the virtual node is not used as a worm hole for routing.
            weight = 10000000 + (y - tech.unit_cell_height // 2) // tech.routing_grid_pitch_y
            G.add_edge(virtual_net_terminal, n, weight=weight)

    return virtual_terminal_nodes
