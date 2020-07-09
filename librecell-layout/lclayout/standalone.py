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
from itertools import chain
from collections import Counter, defaultdict
import numpy
import toml

from lccommon import net_util
from lccommon.net_util import load_transistor_netlist, is_ground_net, is_supply_net
from lclayout.data_types import Cell

from .place.place import TransistorPlacer
from .place.euler_placer import EulerPlacer, HierarchicalPlacer
from .place.smt_placer import SMTPlacer

from .graphrouter.graphrouter import GraphRouter
from .graphrouter.hv_router import HVGraphRouter
from .graphrouter.pathfinder import PathFinderGraphRouter
from .graphrouter.signal_router import DijkstraRouter, ApproxSteinerTreeRouter

from .layout.transistor import *
from .layout import cell_template
from .layout.notch_removal import fill_notches

from .routing_graph import *

from .drc_cleaner import drc_cleaner
from .lvs import lvs

# klayout.db should not be imported if script is run from KLayout GUI.
if 'pya' not in sys.modules:
    import klayout.db as pya

logger = logging.getLogger(__name__)


def _merge_all_layers(shapes):
    # TODO: Move into LcLayout class.
    # Merge all polygons on all layers.
    for layer_name, s in shapes.items():
        if '_label' not in layer_name:
            r = pya.Region(s)
            r.merge()
            s.clear()
            s.insert(r)


def _draw_label(shapes, layer, pos: Tuple[int, int], text: str) -> None:
    """
    Insert a pya.Text object into `shapes`.
    :param shapes:
    :param layer:
    :param pos: Position of the text as a (x,y)-tuple.
    :param text: Text.
    :return: None
    """
    x, y = pos
    # shapes[layer].insert(pya.Text.new(text, pya.Trans(x, y), 0.1, 2))
    shapes[layer].insert(pya.Text.new(text, x, y))


def _draw_routing_tree(shapes: Dict[str, pya.Shapes],
                       G: nx.Graph,
                       rt: nx.Graph,
                       tech,
                       debug_routing_graph: bool = False):
    """ Draw a routing graph into a layout.
    :param shapes: Mapping from layer name to pya.Shapes object
    :param G: Full graph of routing grid
    :param rt: Graph representing the wires
    :param tech: module containing technology information
    :param debug_routing_graph: Draw narrower wires for easier visual inspection
    :return:
    """

    def is_virtual_node(n):
        return n[0].startswith('virtual')

    def is_virtual_edge(e):
        return is_virtual_node(e[0]) or is_virtual_node(e[1])

    logger.debug("Drawing wires")

    # Loop through all edges of the routing tree and draw them individually.
    for a, b in rt.edges:

        if not is_virtual_edge((a, b)):

            l1, (x1, y1) = a
            l2, (x2, y2) = b

            data = G[a][b]

            if l1 == l2:
                # On the same layer -> wire

                w = data.get('wire_width', tech.wire_width[l1])

                ext = w // 2

                is_horizontal = y1 == y2 and x1 != x2

                if is_horizontal:
                    w = tech.wire_width_horizontal[l1]

                if debug_routing_graph:
                    w = min(tech.routing_grid_pitch_x, tech.routing_grid_pitch_y) // 16

                path = pya.Path([pya.Point(x1, y1), pya.Point(x2, y2)], w, ext, ext)
                shapes[l1].insert(path)
            else:
                # l1 != l1 -> this looks like a via
                assert x1 == x2
                assert y1 == y2
                # Draw via
                via_layer = via_layers[l1][l2]['layer']
                logger.debug('Draw via: {} ({}, {})'.format(via_layer, x1, y1))

                via_width = tech.via_size[via_layer]

                if debug_routing_graph:
                    via_width = min(tech.routing_grid_pitch_x, tech.routing_grid_pitch_y) // 16

                w = via_width // 2
                via = pya.Box(pya.Point(x1 - w, y1 - w),
                              pya.Point(x1 + w, y1 + w))
                shapes[via_layer].insert(via)

                # Ensure minimum via enclosure.
                if not debug_routing_graph:
                    for l in (l1, l2):
                        # TODO: Check on which sides minimum enclosure is not yet satisfied by some wire.

                        neighbors = rt.neighbors((l, (x1, y1)))
                        neighbors = [n for n in neighbors if n[0] == l]

                        w_ext = via_width // 2 + tech.minimum_enclosure[(l, via_layer)]
                        w_noext = via_width // 2

                        # Check on which sides the enclosure must be extended.
                        # Some sides will already be covered by a routing wire.
                        ext_right = w_ext
                        ext_upper = w_ext
                        ext_left = w_ext
                        ext_lower = w_ext
                        # TODO
                        # for _, (n_x, n_y) in neighbors:
                        #     if n_x == x1:
                        #         if n_y < y1:
                        #             ext_lower = w_noext
                        #         if n_y > y1:
                        #             ext_upper = w_noext
                        #     if n_y == y1:
                        #         if n_x < x1:
                        #             ext_left = w_noext
                        #         if n_x > x1:
                        #             ext_right = w_noext

                        enc = pya.Box(
                            pya.Point(x1 - ext_left, y1 - ext_lower),
                            pya.Point(x1 + ext_right, y1 + ext_upper)
                        )
                        shapes[l].insert(enc)


def _is_virtual_node_fn(n) -> bool:
    """
    Check if the node is virtual and has no direct physical representation.
    :param n:
    :return:
    """
    return n[0].startswith('virtual')


def _is_virtual_edge_fn(e) -> bool:
    """
    Check if the edge connects to at least one virtual node.
    :param n:
    :return:
    """
    a, b = e
    return _is_virtual_node_fn(a) or _is_virtual_node_fn(b)


class LcLayout:

    def __init__(self,
                 tech,
                 layout: pya.Layout,
                 placer: TransistorPlacer,
                 router: GraphRouter,
                 debug_routing_graph: bool = False,
                 debug_smt_solver: bool = False
                 ):
        assert isinstance(layout, pya.Layout)
        assert isinstance(placer, TransistorPlacer)
        assert isinstance(router, GraphRouter)

        self.tech = tech
        self.layout = layout
        self.placer = placer
        self.router = router
        self.debug_routing_graph = debug_routing_graph
        self.debug_smt_solver = debug_smt_solver

        self.cell_name = None
        self.io_pins = None
        self.SUPPLY_VOLTAGE_NET = None
        self.GND_NET = None

        # Top layout cell.
        self.top_cell: pya.Cell = None

        self._transistors_abstract: List[Transistor] = None

        self._transistor_layouts: Dict[Transistor, TransistorLayout] = None

        self.shapes: Dict[str, db.Shapes] = dict()

        self._routing_terminal_debug_layers = None

        self._abstract_cell: Cell = None
        self._cell_width = None
        self._cell_height = None

        self._spacing_graph = None

        # Routing graph.
        self._routing_graph: nx.Graph = None

        self._routing_trees = None

        # Pin definitions.
        self._pin_shapes = defaultdict(list)

    def _00_00_check_tech(self):
        # Assert that the layers in the keys of multi_via are ordered.
        for l1, l2 in self.tech.multi_via.keys():
            assert l1 <= l2, Exception('Layers must be ordered alphabetically. (%s <= %s)' % (l1, l2))

    def _00_01_prepare_tech(self):
        # Load spacing rules in form of a graph.
        self._spacing_graph = tech_util.spacing_graph(self.tech.min_spacing)

    def _01_load_netlist(self, netlist_path: str, cell_name: str):
        # Load netlist of cell.

        logger.info(f'Load netlist: {netlist_path}')

        self.cell_name = cell_name

        self._transistors_abstract, cell_pins = load_transistor_netlist(netlist_path, cell_name)
        self.io_pins = net_util.get_io_pins(cell_pins)

        ground_nets = {p for p in cell_pins if is_ground_net(p)}
        supply_nets = {p for p in cell_pins if is_supply_net(p)}

        assert len(ground_nets) > 0, "Could not find net name of ground."
        assert len(supply_nets) > 0, "Could not find net name of supply voltage."
        assert len(ground_nets) == 1, "Multiple ground net names: {}".format(ground_nets)
        assert len(supply_nets) == 1, "Multiple supply net names: {}".format(supply_nets)

        self.SUPPLY_VOLTAGE_NET = supply_nets.pop()
        self.GND_NET = ground_nets.pop()

        logger.info("Supply net: {}".format(self.SUPPLY_VOLTAGE_NET))
        logger.info("Ground net: {}".format(self.GND_NET))

        # Convert transistor dimensions into data base units.
        for t in self._transistors_abstract:
            t.channel_width = t.channel_width / self.tech.db_unit

        # Size transistor widths.
        logging.debug('Rescale transistors.')
        for t in self._transistors_abstract:
            t.channel_width = t.channel_width * self.tech.transistor_channel_width_sizing

            min_size = self.tech.minimum_gate_width_nfet if t.channel_type == ChannelType.NMOS else self.tech.minimum_gate_width_pfet

            if t.channel_width + 1e-12 < min_size:
                logger.warning("Channel width too small changing it to minimal size: %.2e < %.2e", t.channel_width,
                               min_size)

            t.channel_width = max(min_size, t.channel_width)

    def _02_setup_layout(self):
        logger.debug("Setup layout.")
        self.top_cell = self.layout.create_cell(self.cell_name)

        # Setup layers.
        self.shapes = dict()
        for name, (num, purpose) in layermap.items():
            layer = self.layout.layer(num, purpose)
            self.shapes[name] = self.top_cell.shapes(layer)

        if self.debug_routing_graph:
            # Layers for displaying routing terminals.
            self._routing_terminal_debug_layers = {
                l: self.layout.layer(idx, 200) for l, (idx, _) in layermap.items()
            }

    def _03_place_transistors(self):
        # Place transistors
        logging.info('Find transistor placement')

        abstract_cell = self.placer.place(self._transistors_abstract)
        logger.info(f"Cell placement:\n\n{abstract_cell}\n")

        self._abstract_cell = abstract_cell

    def _04_draw_transistors(self):
        logger.debug("Draw transistors.")
        # Get the locations of the transistors.
        transistor_locations = self._abstract_cell.get_transistor_locations()

        # Create the layouts of the single transistors. Layouts are already translated to the absolute position.
        self._transistor_layouts = {t: DefaultTransistorLayout(t, (x, y), self.tech)
                                    for t, (x, y) in transistor_locations}

        # Draw the transistors
        for l in self._transistor_layouts.values():
            assert isinstance(l, TransistorLayout)
            l.draw(self.shapes)

    def _05_draw_cell_template(self):
        logger.debug("Draw cell template.")

        tech = self.tech

        # Calculate dimensions of cell.
        num_unit_cells = self._abstract_cell.width
        self._cell_width = (num_unit_cells + 1) * tech.unit_cell_width
        self._cell_height = tech.unit_cell_height

        # Draw cell template.
        cell_template.draw_cell_template(self.shapes,
                                         cell_shape=(self._cell_width, self._cell_height),
                                         nwell_pwell_spacing=self._spacing_graph[l_nwell][l_pwell]['min_spacing']
                                         )

        # Draw power rails.
        vdd_rail = pya.Path([pya.Point(0, tech.unit_cell_height), pya.Point(self._cell_width, tech.unit_cell_height)],
                            tech.power_rail_width)
        vss_rail = pya.Path([pya.Point(0, 0), pya.Point(self._cell_width, 0)], tech.power_rail_width)

        # Insert power rails into layout.
        self.shapes[tech.power_layer].insert(vdd_rail).set_property('net', self.SUPPLY_VOLTAGE_NET)
        self.shapes[tech.power_layer].insert(vss_rail).set_property('net', self.GND_NET)

        # Add pin shapes for power rails.
        self.shapes[tech.pin_layer + '_pin'].insert(vdd_rail)
        self.shapes[tech.pin_layer + '_pin'].insert(vss_rail)

        # Register Pins/Ports for LEF file.
        self._pin_shapes[self.SUPPLY_VOLTAGE_NET].append((tech.power_layer, vdd_rail))
        self._pin_shapes[self.GND_NET].append((tech.power_layer, vss_rail))

    def _06_route(self):

        # TODO: Move as much as possible of the grid construction into a router specific class.
        tech = self.tech

        # Construct two dimensional grid which defines the routing graph on a single layer.
        grid = Grid2D((tech.grid_offset_x, tech.grid_offset_y),
                      (
                          tech.grid_offset_x + self._cell_width - tech.grid_offset_x,
                          tech.grid_offset_y + tech.unit_cell_height),
                      (tech.routing_grid_pitch_x, tech.routing_grid_pitch_y))

        # Create base graph
        G = create_routing_graph_base(grid, tech)

        # Remove illegal routing nodes from graph and get a dict of legal routing nodes per layer.
        remove_illegal_routing_edges(G, self.shapes, tech)

        # if not debug_routing_graph:
        #     assert nx.is_connected(G)

        # Remove pre-routed edges from G.
        remove_existing_routing_edges(G, self.shapes, tech)

        # Create a list of terminal areas: [(net, layer, [terminal, ...]), ...]
        terminals_by_net = extract_terminal_nodes(G, self.shapes, tech)

        # Embed transistor terminal nodes in to routing graph.
        embed_transistor_terminal_nodes(G, terminals_by_net, self._transistor_layouts, tech)

        # Remove terminals of nets with only one terminal. They need not be routed.
        # This can happen if a net is already connected by abutment of two transistors.
        # Count terminals of a net.
        num_appearance = Counter(chain((net for net, _, _ in terminals_by_net), self.io_pins))
        terminals_by_net = [t for t in terminals_by_net if num_appearance[t[0]] > 1]

        # Check if each net really has a routing terminal.
        # It can happen that there is none due to spacing issues.
        # First find all net names in the layout.
        all_net_names = {s.property('net') for _, _shapes in self.shapes.items() for s in _shapes.each()}
        all_net_names -= {None}

        error = False
        # Check if each net has at least a terminal.
        for net_name in all_net_names:
            num_terminals = num_appearance.get(net_name)
            if num_terminals is None or num_terminals == 0:
                logger.error("Net '{}' has no routing terminal.".format(net_name))
                error = True

        if not self.debug_routing_graph:
            assert not error, "Nets without terminals. Check the routing graph (--debug-routing-graph)!"

        # Create virtual graph nodes for each net terminal.
        virtual_terminal_nodes = create_virtual_terminal_nodes(G, terminals_by_net, self.io_pins, tech)

        if self.debug_routing_graph:
            # Display terminals on layout.
            routing_terminal_shapes = {
                l: self.top_cell.shapes(self._routing_terminal_debug_layers[l]) for l in tech.routing_layers.keys()
            }
            for net, layer, ts in terminals_by_net:
                for x, y in ts:
                    d = tech.routing_grid_pitch_x // 16
                    routing_terminal_shapes[layer].insert(pya.Box(pya.Point(x - d, y - d), pya.Point(x + d, y + d)))

        # Remove nodes that will not be used for routing.
        # Iteratively remove nodes of degree 1.
        while True:
            unused_nodes = set()
            for n in G:
                if nx.degree(G, n) <= 1:
                    if not _is_virtual_node_fn(n):
                        unused_nodes.add(n)
            if len(unused_nodes) == 0:
                break
            G.remove_nodes_from(unused_nodes)

        if not nx.is_connected(G):
            assert False, 'Routing graph is not connected.'

        self._routing_graph = G

        # TODO: SPLIT HERE
        # def _07_route(self):

        tech = self.tech
        spacing_graph = self._spacing_graph
        G = self._routing_graph

        # Route
        if self.debug_routing_graph:
            # Write the full routing graph to GDS.
            logger.info("Skip routing and plot routing graph.")
            self._routing_trees = {'graph': self._routing_graph}
        else:
            logger.info("Start routing")
            # For each routing node find other nodes that are close enough that they cannot be used
            # both for routing. This is used to avoid spacing violations during routing.
            logger.debug("Find conflicting nodes.")
            conflicts = dict()
            # Loop through all nodes in the routing graph G.
            for n in G:
                # Skip virtual nodes which have no physical representation.
                if not _is_virtual_node_fn(n):
                    layer, point = n
                    wire_width1 = tech.wire_width.get(layer, 0) // 2
                    node_conflicts = set()
                    if layer in spacing_graph:
                        # If there is a spacing rule defined involving `layer` then
                        # loop through all layers that have a spacing rule defined
                        # relative to the layer of the current node n.
                        for other_layer in spacing_graph[layer]:
                            if other_layer in tech.routing_layers:
                                # Find minimal spacing of nodes such that spacing rule is asserted.
                                wire_width2 = tech.wire_width.get(other_layer, 0) // 2
                                min_spacing = spacing_graph[layer][other_layer]['min_spacing']
                                margin = (wire_width1 + wire_width2 + min_spacing)

                                # Find nodes that are closer than the minimal spacing.
                                # conflict_points = grid.neigborhood(point, margin, norm_ord=1)
                                potential_conflicts = [x for x in G if x[0] == other_layer]
                                conflict_points = [p for (_, p) in potential_conflicts
                                                   if numpy.linalg.norm(numpy.array(p) - numpy.array(point),
                                                                        ord=1) < margin
                                                   ]
                                # Construct the lookup table for conflicting nodes.
                                for p in conflict_points:
                                    conflict_node = other_layer, p
                                    if conflict_node in G:
                                        node_conflicts.add(conflict_node)
                    if node_conflicts:
                        conflicts[n] = node_conflicts

            # Find routing nodes that are reserved for a net. They cannot be used to route other nets.
            # (For instance the ends of a gate stripe.)
            reserved_nodes = defaultdict(set)
            for net, layer, terminals in terminals_by_net:
                for p in terminals:
                    n = layer, p
                    reserved = reserved_nodes[net]
                    reserved.add(n)
                    if n in conflicts:
                        for c in conflicts[n]:  # Also reserve nodes that would cause a spacing violation.
                            reserved.add(c)

            assert nx.is_connected(G)

            # Invoke router and store result.
            self._routing_trees = self.router.route(G,
                                                    signals=virtual_terminal_nodes,
                                                    reserved_nodes=reserved_nodes,
                                                    node_conflict=conflicts,
                                                    is_virtual_node_fn=_is_virtual_node_fn
                                                    )

            # TODO: Sanity check on result.

    def _08_draw_routes(self):
        # Draw the layout of the routes.
        for signal_name, rt in self._routing_trees.items():
            _draw_routing_tree(self.shapes, self._routing_graph, rt, self.tech, self.debug_routing_graph)

        # Merge the polygons on all layers.
        _merge_all_layers(self.shapes)

    def _09_post_process(self):
        tech = self.tech
        # Register Pins/Ports for LEF file.

        if not self.debug_routing_graph:

            # Clean DRC violations that are not handled above.

            def fill_all_notches():
                # Remove notches on all layers.
                for layer, s in self.shapes.items():
                    if layer in tech.minimum_notch:

                        if layer in tech.connectable_layers:
                            r = pya.Region(s)
                            filled = fill_notches(r, tech.minimum_notch[layer])
                            s.insert(filled)
                        else:
                            # Remove notches per polygon to avoid connecting independent shapes.
                            s_filled = pya.Shapes()
                            for shape in s.each():
                                r = pya.Region(shape.polygon)

                                filled = fill_notches(r, tech.minimum_notch[layer])
                                s_filled.insert(filled)

                            s.insert(s_filled)

                    _merge_all_layers(self.shapes)

            # Fill notches that violate a notch rule.
            fill_all_notches()
            # Do a second time because first iteration could have introduced new notch violations.
            fill_all_notches()

            # Fix minimum area violations.
            fix_min_area(tech, self.shapes, debug=self.debug_smt_solver)

            # Draw pins
            # Get shapes of pins.
            pin_locations_by_net = {}
            pin_shapes_by_net = {}
            for net_name, rt in self._routing_trees.items():
                # Get virtual pin nodes.
                virtual_pins = [n for n in rt.nodes if n[0] == 'virtual_pin']
                for vp in virtual_pins:
                    # Get routing nodes adjacent to virtual pin nodes. They contain the location of the pin.
                    locations = [l for _, l in rt.edges(vp)]
                    _, net_name, _, _ = vp
                    for layer, (x, y) in locations:
                        w = tech.minimum_pin_width
                        s = self.shapes[layer]

                        # Find shape at (x,y).
                        ball = pya.Box(pya.Point(x - 1, y - 1), pya.Point(x + 1, y + 1))
                        pin_shapes = pya.Region(s).interacting(pya.Region(ball))

                        # Remember pin location and shape.
                        pin_locations_by_net[net_name] = x, y
                        pin_shapes_by_net[net_name] = pin_shapes

                        # Register pin shapes for LEF file.
                        self._pin_shapes[net_name].append((layer, pin_shapes))

            # Add pin labels
            for net_name, (x, y) in pin_locations_by_net.items():
                logger.debug('Add pin label: %s, (%d, %d)', net_name, x, y)
                _draw_label(self.shapes, tech.pin_layer + '_label', (x, y), net_name)

            # Add label for power rails
            _draw_label(self.shapes, tech.power_layer + '_label', (self._cell_width // 2, 0), self.GND_NET)
            _draw_label(self.shapes, tech.power_layer + '_label', (self._cell_width // 2, self._cell_height),
                        self.SUPPLY_VOLTAGE_NET)

            # Add pin shapes.
            for net_name, pin_shapes in pin_shapes_by_net.items():
                self.shapes[tech.pin_layer + '_pin'].insert(pin_shapes)

    def create_cell_layout(self, cell_name: str, netlist_path: str) \
            -> Tuple[pya.Cell, Dict[str, List[Tuple[str, pya.Shape]]]]:

        self._00_00_check_tech()
        self._00_01_prepare_tech()

        self._01_load_netlist(netlist_path, cell_name)

        self._02_setup_layout()
        self._03_place_transistors()
        self._04_draw_transistors()
        self._05_draw_cell_template()
        self._06_route()
        self._08_draw_routes()
        self._09_post_process()

        return self.top_cell, self._pin_shapes


def main():
    """
    Entry function for standalone command line tool.
    :return:
    """
    import argparse
    import datetime
    import time

    # List of available placer engines.
    placers = {
        'flat': EulerPlacer,
        'hierarchical': HierarchicalPlacer,
        'smt': SMTPlacer
    }

    signal_routers = {
        'dijkstra': DijkstraRouter,  # Fast but not stable.
        'steiner': ApproxSteinerTreeRouter,  # Slow but best results.
        # 'lp': LPSignalRouter
    }

    # Define commandline arguments.
    parser = argparse.ArgumentParser(description='Generate GDS layout from SPICE netlist.')
    parser.add_argument('--cell', required=True, metavar='NAME', type=str, help='cell name')
    parser.add_argument('--netlist', required=True, metavar='FILE', type=str, help='path to SPICE netlist')
    parser.add_argument('--output-dir', default='.', metavar='DIR', type=str, help='output directory for layouts')
    parser.add_argument('--tech', required=True, metavar='FILE', type=str, help='technology file')

    parser.add_argument('--debug-routing-graph', action='store_true',
                        help='write full routing graph to the layout instead of wires')
    parser.add_argument('--debug-smt-solver', action='store_true',
                        help='enable debug mode: display routing nodes in layout, \
                        show unsatisfiable core if SMT DRC cleaning fails.')

    parser.add_argument('--placer', default='flat', metavar='PLACER', type=str, choices=placers.keys(),
                        help='placement algorithm ({})'.format(', '.join(sorted(placers.keys()))))

    parser.add_argument('--signal-router', default='dijkstra', metavar='SIGNAL_ROUTER', type=str,
                        choices=signal_routers.keys(),
                        help='routing algorithm for single signals ({})'.format(
                            ', '.join(sorted(signal_routers.keys()))))

    # parser.add_argument('--profile', action='store_true', help='enable profiler')
    parser.add_argument('-v', '--verbose', action='store_true', help='show more information')
    parser.add_argument('--ignore-lvs', action='store_true', help='Write the layout file even if the LVS check failed.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="don't show any information except fatal events (overwrites --verbose)")
    parser.add_argument('--log', required=False, metavar='LOG_FILE', type=str,
                        help='write log to this file instead of stdout')

    # Parse arguments
    args = parser.parse_args()

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    if args.quiet:
        log_level = logging.FATAL

    # Setup logging
    logging.basicConfig(format='%(asctime)s %(module)16s %(levelname)8s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=log_level,
                        filename=args.log)

    # Load netlist of cell
    cell_name = args.cell
    netlist_path = args.netlist

    tech_file = args.tech

    tech = tech_util.load_tech_file(tech_file)

    # Create empty layout
    layout = pya.Layout()

    # Setup placer algorithm

    placer = placers[args.placer]()
    logger.info("Placement algorithm: {}".format(type(placer).__name__))

    # Setup routing algorithm

    signal_router = signal_routers[args.signal_router]()
    logger.info("Signal routing algorithm: {}".format(type(signal_router).__name__))

    router = PathFinderGraphRouter(
        signal_router
    )
    # router = LPGraphRouter()
    router = HVGraphRouter(router,
                           orientation_change_penalty=tech.orientation_change_penalty
                           )

    layouter = LcLayout(tech=tech,
                        layout=layout,
                        placer=placer,
                        router=router,
                        debug_routing_graph=args.debug_routing_graph,
                        debug_smt_solver=args.debug_smt_solver)

    # Run layout synthesis
    time_start = time.process_time()
    cell, pin_geometries = layouter.create_cell_layout(cell_name, netlist_path)

    # LVS check
    logger.info("Running LVS check")
    reference_netlist = lvs.read_netlist_mos4_to_mos3(netlist_path)

    # Remove all unused circuits.
    # The reference netlist must contain only the circuit of the cell to be checked.
    # Copying a circuit into a new netlist makes `combine_devices` fail.
    circuits_to_delete = {c for c in reference_netlist.each_circuit() if c.name != cell_name}
    for c in circuits_to_delete:
        reference_netlist.remove(c)

    # Extract netlist from layout.
    extracted_netlist = lvs.extract_netlist(layout, cell)

    # Run LVS comparison of the two netlists.
    lvs_success = lvs.compare_netlist(extracted_netlist, reference_netlist)

    logger.info("LVS result: {}".format('SUCCESS' if lvs_success else 'FAILED'))

    if not lvs_success:
        logger.error("LVS check failed!")
        if not args.ignore_lvs and not args.debug_routing_graph:
            exit(1)

    # Output using defined output writers.
    from .writer.writer import Writer
    for writer in tech.output_writers:
        assert isinstance(writer, Writer)
        logger.debug("Call output writer: {}".format(type(writer).__name__))
        writer.write_layout(
            layout=layout,
            pin_geometries=pin_geometries,
            top_cell=cell,
            output_dir=args.output_dir
        )

    time_end = time.process_time()
    duration = datetime.timedelta(seconds=time_end - time_start)
    logger.info("Done (Total duration: {})".format(duration))


def fix_min_area(tech, shapes: Dict[str, pya.Shapes], debug=False):
    """
    Fix minimum area violations.
    This is a wrapper around the drc_cleaner module.
    :param tech:
    :param shapes:
    :param debug: Tell DRC cleaner to find unsatisiable core.
    :return:
    """

    # Find minimum area violations.
    # And create a set of whitelisted shapes that are allowed to be changed for DRC cleaning.
    min_area_violations = set()
    for layer, _shapes in shapes.items():
        min_area = tech.min_area.get(layer, 0)
        for shape in _shapes.each():
            area = shape.area()
            if area < min_area:
                min_area_violations.add((layer, shape))

    # TODO: Also whitelist vias connected to the violating shapes.

    if min_area_violations:
        success = drc_cleaner.clean(tech,
                                    shapes=shapes,
                                    white_list=min_area_violations,
                                    enable_min_area=True,
                                    debug=debug
                                    )
        if not success:
            logger.error("Minimum area fixing failed!")
    else:
        logger.info("No minimum area violations.")
