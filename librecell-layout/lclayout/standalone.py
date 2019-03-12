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
from itertools import chain
from collections import Counter
import numpy

from lccommon import net_util
from lccommon.net_util import load_transistor_netlist, is_ground_net, is_supply_net
from .place.place import TransistorPlacer
from .place.euler_placer import EulerPlacer, HierarchicalPlacer
from .place.smt_placer import SMTPlacer

from .graphrouter.hv_router import HVGraphRouter
from .graphrouter.pathfinder import PathFinderGraphRouter
from .graphrouter.signal_router import DijkstraRouter

from .layout.transistor import *
from .layout.notch_removal import fill_notches
from .lef import types as lef

from .routing_graph import *

from .drc_cleaner import drc_cleaner

# klayout.db should not be imported if script is run from KLayout GUI.
if 'pya' not in sys.modules:
    import klayout.db as pya

logger = logging.getLogger(__name__)


def _merge_all_layers(shapes):
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
    shapes[layer].insert(pya.Text.new(text, pya.Trans(x, y), 0.1, 2))


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
    :param debug_routing_graph: Draw narrowser wires for easier visual inspection
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

                via_width = tech.via_size[via_layer]

                if debug_routing_graph:
                    via_width = min(tech.routing_grid_pitch_x, tech.routing_grid_pitch_y) // 16

                w = via_width // 2
                via = pya.Box(pya.Point(x1 - w, y1 - w),
                              pya.Point(x1 + w, y1 + w))
                shapes[via_layer].insert(via)

                # Ensure minimum via enclosure.
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


def create_cell_layout(tech, layout: pya.Layout, cell_name: str, netlist_path: str,
                       placer: TransistorPlacer = None,
                       debug_routing_graph: bool = False,
                       debug_smt_solver: bool = False) -> Tuple[pya.Cell, Dict[str, Tuple[str, pya.Shape]]]:
    """ Draw the layout of a cell.

    Parameters
    ----------
    :param tech: module containing technology information
    :param layout: klayout.db.Layout
    :param cell_name: str
      The name of the cell to be drawn.
    :param netlist_path: Path to SPICE transistor netlist.
    :param debug_routing_graph: bool
      If set to True, the full routing graph is written to the layout instead of the routing paths.
    :param debug_smt_solver: Tell DRC cleaner to show which assertions are not satisfiable.
    :return Returns the new pya.Cell and a Dict containing the pin shapes for each pin name.
        (cell, {net_name: (layer_name, pya.Shape)})
    """

    # Load netlist of cell
    logger.info('Load netlist: %s', netlist_path)
    transistors_abstract, cell_pins = load_transistor_netlist(netlist_path, cell_name)
    io_pins = net_util.get_io_pins(cell_pins)

    top = layout.create_cell(cell_name)

    # Setup layers.
    shapes = {}
    for name, (num, purpose) in layermap.items():
        layer = layout.layer(num, purpose)
        shapes[name] = top.shapes(layer)

    if debug_routing_graph:
        # Layers for displaying routing terminals.
        routing_terminal_debug_layers = {
            l: layout.layer(idx, 200) for l, (idx, _) in layermap.items()
        }

    # Assert that the layers in the keys of multi_via are ordered.
    for l1, l2 in tech.multi_via.keys():
        assert l1 <= l2, Exception('Layers must be ordered alphabetically. (%s <= %s)' % (l1, l2))

    # Size transistor widths.
    logging.debug('Rescale transistors.')
    for t in transistors_abstract:
        t.channel_width = t.channel_width * tech.transistor_channel_width_sizing

        min_size = tech.minimum_gate_width_nfet if t.channel_type == ChannelType.NMOS else tech.minimum_gate_width_pfet
        min_size *= tech.db_unit

        if t.channel_width < min_size:
            logger.warning("Channel width too small changing it to minimal size: %.2e < %.2e", t.channel_width,
                           min_size)
            t.channel_width = min_size

    # Place transistors
    logging.info('Find transistor placement')
    if placer is None:
        logging.debug('No placer defined. Using `EulerPlacer`.')
        placer = EulerPlacer()
    abstract_cell = placer.place(transistors_abstract)
    print(abstract_cell)

    # Calculate dimensions of cell.
    num_unit_cells = abstract_cell.width
    cell_width = (num_unit_cells + 1) * tech.unit_cell_width
    cell_height = tech.unit_cell_height

    # Get the locations of the transistors.
    transistors = abstract_cell.get_transistor_locations()

    # Create the layouts of the single transistors. Layouts are already translated to the absolute position.
    transistor_layouts = {t: create_transistor_layout(t, (x, y), tech) for t, (x, y) in transistors}

    # Draw the transistors
    for l in transistor_layouts.values():
        draw_transistor(l, shapes)

    # Create mapping from nets to {layer: region}
    net_regions = {}

    # Draw cell template.
    # Draw abutment box.
    shapes[l_abutment_box].insert(pya.Box(0, 0, cell_width, cell_height))

    # Draw power rails.
    vdd_rail = pya.Path([pya.Point(0, tech.unit_cell_height), pya.Point(cell_width, tech.unit_cell_height)],
                        tech.power_rail_width)
    vss_rail = pya.Path([pya.Point(0, 0), pya.Point(cell_width, 0)], tech.power_rail_width)

    shapes[tech.power_layer].insert(vdd_rail)
    shapes[tech.power_layer].insert(vss_rail)

    ground_nets = {p for p in cell_pins if is_ground_net(p)}
    supply_nets = {p for p in cell_pins if is_supply_net(p)}

    assert len(ground_nets) > 0, "Could not find net name of ground."
    assert len(supply_nets) > 0, "Could not find net name of supply voltage."
    assert len(ground_nets) == 1, "Multiple ground net names: {}".format(ground_nets)
    assert len(supply_nets) == 1, "Multiple supply net names: {}".format(supply_nets)

    SUPPLY_VOLTAGE_NET = supply_nets.pop()
    GND_NET = ground_nets.pop()

    logger.info("Supply net: {}".format(SUPPLY_VOLTAGE_NET))
    logger.info("Ground net: {}".format(GND_NET))

    # Register power rails as net regions.
    for net, shape in [(SUPPLY_VOLTAGE_NET, vdd_rail), (GND_NET, vss_rail)]:
        net_regions.setdefault(net, {}).setdefault(tech.power_layer, pya.Region()).insert(shape)

    # Register Pins/Ports for LEF file.
    lef_ports = {}
    lef_ports.setdefault(SUPPLY_VOLTAGE_NET, []).append((tech.power_layer, vdd_rail))
    lef_ports.setdefault(GND_NET, []).append((tech.power_layer, vss_rail))

    # Pre-route vertical gate-gate connections
    for i in range(abstract_cell.width):
        u = abstract_cell.upper[i]
        l = abstract_cell.lower[i]

        if u is not None and l is not None:
            if u.gate == l.gate:
                logger.debug("Pre-route gate at x position %d", i)
                tu = transistor_layouts[u]
                tl = transistor_layouts[l]

                a = tu.gate.bbox().center()
                b = tl.gate.bbox().center()
                # Create gate shape.
                gate_path = pya.Path.new(
                    [a, b],
                    tech.gate_length)

                shapes[l_poly].insert(gate_path)
                net_regions.setdefault(u.gate, {}).setdefault(l_poly, pya.Region()).insert(gate_path)
                # tu.terminals.clear()
                # tl.terminals.clear()

    # Construct net regions of transistors.
    for a, l in transistor_layouts.items():
        net_shapes = [
            # (l_poly, a.gate, l.gate),
            (l_active, a.left, l.source_box),
            (l_active, a.right, l.drain_box)
        ]

        for layer, net, shape in net_shapes:
            r = net_regions.setdefault(net, {}).setdefault(layer, pya.Region())
            r.insert(shape)
            r.merge()

    grid = Grid2D((tech.grid_offset_x, tech.grid_offset_y),
                  (tech.grid_offset_x + cell_width - tech.grid_offset_x, tech.grid_offset_y + tech.unit_cell_height),
                  (tech.routing_grid_pitch_x, tech.routing_grid_pitch_y))

    # Create base graph
    G = create_routing_graph_base(grid, tech)

    # Remove illegal routing nodes from graph and get a dict of legal routing nodes per layer.
    routing_nodes = prepare_routing_nodes(G, grid, shapes, tech)

    # Remove pre-routed edges from G.
    remove_existing_routing_edges(G, shapes, tech)

    # Create a list of terminal areas: [(net, layer, [terminal, ...]), ...]
    terminals_by_net = extract_terminal_nodes(routing_nodes, net_regions, tech)

    # Embed transistor terminal nodes in to routing graph.
    embed_transistor_terminal_nodes(G, terminals_by_net, transistor_layouts, tech)

    # Remove terminals of nets with only one terminal. They need not be routed.
    # This can happen if a net is already connected by abutment of two transistors.
    # Count terminals of a net.
    num_appearance = Counter(chain((net for net, _, _ in terminals_by_net), io_pins))
    terminals_by_net = [t for t in terminals_by_net if num_appearance[t[0]] > 1]

    # Check if each net really has a routing terminal.
    # It can happen that there is none due to spacing issues.
    error = False
    for net_name, layer, terminals in terminals_by_net:
        if len(terminals) == 0:
            logger.error("Net '{}' has no routing terminal.".format(net_name))
            error = True

    if not debug_routing_graph:
        assert not error, "Nets without terminals. Check the routing graph (--debug-routing-graph)!"

    # Create virtual graph nodes for each net terminal.
    virtual_terminal_nodes = create_virtual_terminal_nodes(G, routing_nodes, terminals_by_net, io_pins, tech)

    if debug_routing_graph:
        # Display terminals on layout.
        routing_terminal_shapes = {
            l: top.shapes(routing_terminal_debug_layers[l]) for l in tech.routing_layers.keys()
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
                if not n[0].startswith('virtual'):
                    unused_nodes.add(n)
        if len(unused_nodes) == 0:
            break
        G.remove_nodes_from(unused_nodes)

    if not nx.is_connected(G):
        assert False, 'Routing graph is not connected.'

    # Route
    if debug_routing_graph:
        # Write the full routing graph to GDS.
        logger.info("Skip routing and plot routing graph.")
        routing_trees = {'graph': G}
    else:
        logger.info("Start routing")
        '''
        # Setup heuristic for A-Star router
        weight_x = min((w for l,w in tech.weights_horizontal.items() if 'h' in tech.routing_layers[l]))
        weight_y = min((w for l,w in tech.weights_vertical.items() if 'v' in tech.routing_layers[l]))
        
        def heuristic(a, b):
          
          orientation1, a = a
          orientation2, b = b
          
          if a[0].startswith('virtual') or b[0].startswith('virtual'):
            return 0
            
          
          l1, (x1,y1) = a
          l2, (x2,y2) = b
          
          dx = abs(x2-x1)
          dy = abs(y2-y1)
          
          dist = dx*weight_x + dy*weight_y
          
          return dist
        
        detail_router = AStarRouter(heuristic_fn=heuristic)
        '''

        """
        For each routing node find other nodes that are close enough that they cannot be used
        both for routing. This is used to avoid spacing violations during routing.
        """
        logger.debug("Find conflicting nodes.")
        conflicts = dict()
        spacing_graph = tech_util.spacing_graph(tech.min_spacing)
        # Loop through all nodes in the routing graph G.
        for n in G:
            # Skip virtual nodes wich have no physical representation.
            if not n[0].startswith('virtual'):
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
                                                                    ord=1) <= margin
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
        reserved_nodes = dict()
        for net, layer, terminals in terminals_by_net:
            for p in terminals:
                n = layer, p
                reserved = reserved_nodes.setdefault(net, set())
                reserved.add(n)
                if n in conflicts:
                    for c in conflicts[n]:  # Also reserve nodes that would cause a spacing violation.
                        reserved.add(c)

        detail_router = DijkstraRouter()
        # detail_router = LPSignalRouter()
        # detail_router = ApproxSteinerTreeRouter()
        assert nx.is_connected(G)

        router = PathFinderGraphRouter(detail_router)
        # router = LPGraphRouter()
        router = HVGraphRouter(router,
                               orientation_change_penalty=tech.orientation_change_penalty)

        routing_trees = router.route(G,
                                     signals=virtual_terminal_nodes,
                                     reserved_nodes=reserved_nodes,
                                     node_conflict=conflicts
                                     )

    # Draw the layout of the routes.
    for signal_name, rt in routing_trees.items():
        _draw_routing_tree(shapes, G, rt, tech, debug_routing_graph)

    # Merge the polygons on all layers.
    _merge_all_layers(shapes)

    def fill_all_notches():
        # Remove notches on all layers.
        for layer, s in shapes.items():
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

            _merge_all_layers(shapes)

    if not debug_routing_graph:

        # Clean DRC violations that are not handled above.

        # Fill half of the cell with nwell.
        # TODO: do this in the cell template or after placing the transistors.
        nwell_box = pya.Box(
            pya.Point(0, cell_height // 2),
            pya.Point(cell_width, cell_height)
        )

        shapes[l_nwell].insert(nwell_box)

        # Fill notches that violate a notch rule.
        fill_all_notches()
        # Do a second time because first iteration could have introduced new notch violations.
        fill_all_notches()

        # Fix minimum area violations.
        fix_min_area(tech, shapes, debug=debug_smt_solver)

        # Draw pins
        # Get shapes of pins.
        pin_locations_by_net = {}
        pin_shapes_by_net = {}
        for net_name, rt in routing_trees.items():
            # Get virtual pin nodes.
            virtual_pins = [n for n in rt.nodes if n[0] == 'virtual_pin']
            for vp in virtual_pins:
                # Get routing nodes adjacent to virtual pin nodes. They contain the location of the pin.
                locations = [l for _, l in rt.edges(vp)]
                _, net_name, _, _ = vp
                for layer, (x, y) in locations:
                    w = tech.minimum_pin_width
                    s = shapes[layer]

                    # Find shape at (x,y).
                    ball = pya.Box(pya.Point(x - 1, y - 1), pya.Point(x + 1, y + 1))
                    pin_shapes = pya.Region(s).interacting(pya.Region(ball))

                    # Remember pin location and shape.
                    pin_locations_by_net[net_name] = x, y
                    pin_shapes_by_net[net_name] = pin_shapes

                    # Register pin shapes for LEF file.
                    lef_ports.setdefault(net_name, []).append((layer, pin_shapes))

        # Add pin labels
        for net_name, (x, y) in pin_locations_by_net.items():
            logger.debug('Add pin label: %s, (%d, %d)', net_name, x, y)
            _draw_label(shapes, tech.pin_layer + '_label', (x, y), net_name)

        # Add pin shapes.
        for net_name, pin_shapes in pin_shapes_by_net.items():
            shapes[tech.pin_layer + '_pin'].insert(pin_shapes)

        # Add label for power rails
        _draw_label(shapes, tech.power_layer + '_label', (cell_width // 2, 0), GND_NET)
        _draw_label(shapes, tech.power_layer + '_label', (cell_width // 2, cell_height), SUPPLY_VOLTAGE_NET)

    return top, lef_ports


def main():
    """
    Entry function for standalone command line tool.
    :return:
    """
    import argparse
    import os
    import datetime
    import time

    # List of available placer engines.
    placers = {
        'flat': EulerPlacer,
        'hierarchical': HierarchicalPlacer,
        'smt': SMTPlacer
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
                        help='placement algorithm')

    # parser.add_argument('--profile', action='store_true', help='enable profiler')
    parser.add_argument('-v', '--verbose', action='store_true', help='show more information')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="don't show any information except fatal events (overwrites --verbose)")

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
                        level=log_level)

    # Load netlist of cell
    cell_name = args.cell
    path = args.netlist

    tech_file = args.tech

    tech = tech_util.load_tech_file(tech_file)

    # Create empty layout
    layout = pya.Layout()

    # Run layout synthesis
    time_start = time.process_time()
    cell, pin_geometries = create_cell_layout(tech, layout, cell_name, path,
                                              placer=placers[args.placer](),
                                              debug_routing_graph=args.debug_routing_graph,
                                              debug_smt_solver=args.debug_smt_solver)

    def remap_layers(layout: pya.Layout) -> pya.Layout:
        """
        Rename layer to match the scheme defined in the technology file.
        :param layout:
        :return:
        """
        logger.info("Remap layers.")
        layout2 = pya.Layout()
        top1 = layout.top_cell()
        top2 = layout2.create_cell(cell_name)
        layer_infos1 = layout.layer_infos()
        for layer_info in layer_infos1:

            src_layer = (layer_info.layer, layer_info.datatype)

            if src_layer not in layermap_reverse:
                logger.warning("Layer {} not defined in `layermap_reverse`.".format(src_layer))
                dest_layers = src_layer
            else:
                src_layer_name = layermap_reverse[src_layer]
                dest_layers = tech.output_map[src_layer_name]

            if not isinstance(dest_layers, list):
                dest_layers = [dest_layers]

            src_idx = layout.layer(layer_info)
            for dest_layer in dest_layers:
                dest_idx = layout2.layer(*dest_layer)
                top2.shapes(dest_idx).insert(top1.shapes(src_idx))

        return layout2

    # Re-map layers
    layout = remap_layers(layout)

    # Store result
    gds_file_name = '{}.gds'.format(cell_name)
    gds_out_path = os.path.join(args.output_dir, gds_file_name)
    logger.info("Write GDS: %s", gds_out_path)
    layout.write(gds_out_path)

    # LEF output
    if not args.debug_routing_graph:
        # Create and populate LEF Macro data structure.
        # TODO: pass correct USE and DIRECTION
        lef_macro = generate_lef_macro(cell_name,
                                       pin_geometries=pin_geometries,
                                       pin_use=None,
                                       pin_direction=None)

        # Write LEF
        lef_file_name = "{}.lef".format(cell_name)
        lef_output_path = os.path.join(args.output_dir, lef_file_name)

        with open(lef_output_path, "w") as f:
            logger.info("Write LEF: {}".format(lef_output_path))
            f.write(lef.lef_format(lef_macro))

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


def generate_lef_macro(cell_name: str,
                       pin_geometries: Dict[str, List[Tuple[str, pya.Shape]]],
                       pin_direction: Dict[str, lef.Direction],
                       pin_use: Dict[str, lef.Use]
                       ) -> lef.Macro:
    """
    Assemble a LEF MACRO structure containing the pin shapes.
    :param cell_name: Name of the cell as it will appear in the LEF file.
    :param pin_geometries: A dictionary mapping pin names to geometries: Dict[pin name, List[(layer name, klayout Shape)]]
    :param pin_direction:
    :param pin_use:
    :return: Returns a `lef.Macro` object containing the pin information of the cell.

    # TODO: FOREIGN statement (reference to GDS)
    """

    logger.debug("Generate LEF MACRO structure for {}.".format(cell_name))
    pins = []
    # Create LEF Pin objects containing geometry information of the pins.
    for pin_name, ports in pin_geometries.items():

        layers = []

        for layer_name, shape in ports:
            # Convert all non-regions into a region
            region = pya.Region()
            region.insert(shape)
            region.merge()

            geometries = []
            for p in region.each_merged():
                polygon = p.to_simple_polygon()

                box = polygon.bbox()
                is_box = pya.SimplePolygon(box) == polygon

                if is_box:
                    rect = lef.Rect((box.p1.x, box.p1.y), (box.p2.x, box.p2.y))
                    geometries.append(rect)
                else:
                    # Port is a polygon
                    # Convert `pya.Point`s into LEF points.
                    points = [(p.x, p.y) for p in polygon.each_point()]
                    poly = lef.Polygon(points)
                    geometries.append(poly)

            layers.append((lef.Layer(layer_name), geometries))

        port = lef.Port(CLASS=lef.Class.CORE,
                        geometries=layers)

        # if pin_name not in pin_direction:
        #     msg = "I/O direction of pin '{}' is not defined.".format(pin_name)
        #     logger.error(msg)
        #     assert False, msg
        #
        # if pin_name not in pin_use:
        #     msg = "Use of pin '{}' is not defined. Must be one of (CLK, SIGNAL, POWER, ...)".format(pin_name)
        #     logger.error(msg)
        #     assert False, msg

        pin = lef.Pin(pin_name=pin_name,
                      direction=lef.Direction.INOUT,  # TODO: find direction
                      use=lef.Use.SIGNAL,  # TODO: correct use
                      shape=lef.Shape.ABUTMENT,
                      port=port,
                      property={},
                      )
        pins.append(pin)

    macro = lef.Macro(
        name=cell_name,
        macro_class=lef.MacroClass.CORE,
        foreign=lef.Foreign(cell_name, lef.Point(0, 0)),
        origin=lef.Point(0, 0),
        symmetry={lef.Symmetry.X, lef.Symmetry.Y, lef.Symmetry.R90},
        site="CORE",
        pins=pins,
        obstructions=[]
    )

    return macro
