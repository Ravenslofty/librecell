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
from .place import TransistorPlacer
from ..extrema import all_min, all_max
from lccommon import net_util

from ..data_types import *

import numpy as np
from itertools import tee, chain, cycle, islice, permutations, product
from typing import Iterable, Tuple, Hashable, List, Set, Any, Dict

import networkx as nx

from . import eulertours
from . import partition

import logging

logger = logging.getLogger(__name__)


def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


def _trim_none(l: List) -> List:
    """ Cut away all `None`s from the start and end of the list `l`.
    :param l:
    :return:
    """
    while l and l[0] is None:
        l = l[1:]
    while l and l[-1] is None:
        l = l[:-1]
    return l


def _remove_gaps(cell: Cell) -> Cell:
    """
    Remove unnecessary diffusion gaps where possible.
    :param cell: The input cell.
    :return: The compacted.
    """

    upper = []
    lower = []

    n = len(cell.upper)
    assert n == len(cell.lower)

    for i in range(n):
        l = cell.lower[i]
        u = cell.upper[i]

        if u is not None or l is not None:
            upper.append(u)
            lower.append(l)
        else:
            # Both places are empty.
            assert u is None and l is None

            # Place the gap only if it is need to avoid a short circuit.
            if i >= 1 and i + 1 < n:
                prev_u = upper[-1]
                prev_l = lower[-1]
                next_u = cell.upper[i + 1]
                next_l = cell.lower[i + 1]

                # Check if the upper or lower row needs the diffusion gap.
                need_gap = __need_gap(prev_u, next_u) or __need_gap(prev_l, next_l)

                if need_gap:
                    upper.append(None)
                    lower.append(None)

    return _assemble_cell(lower, upper)


def _assemble_cell(lower_row: List[Transistor], upper_row: List[Transistor]) -> Cell:
    """ Build a Cell object from a nmos and pmos row.
    :param lower_row:
    :param upper_row:
    :return:
    """
    width = max(len(lower_row), len(upper_row))
    cell = Cell(width)
    for i, t in enumerate(upper_row):
        cell.upper[i] = t

    for i, t in enumerate(lower_row):
        cell.lower[i] = t
    return cell


def _row_io_ordering_cost(row: List[Transistor], input_nets: Set[Any], output_nets: Set[Any]) -> float:
    """ Return a low cost if input nets are placed on the left side and outputs on the right side.
    """

    nets = [(net, pos) for pos, net in
            enumerate(chain(*[[t.source_net, t.gate_net, t.drain_net] for t in row if t is not None]))
            ]
    return _io_ordering_cost(nets, input_nets, output_nets)


def _io_ordering_cost(nets: Iterable[Tuple[Hashable, float]], input_nets: Set[Any], output_nets: Set[Any]) -> float:
    """ Return a low cost if input nets are placed on the left side and outputs on the right side.
    :param nets: Tuples of (net name, x-position)
    :param input_nets: Names of input nets.
    :param output_nets: Names of output nets.
    :return: Returns a cost which prefers that input nets are placed on the left and output nets are placed on the right.
    """

    nets = list(nets)

    cost = 0
    min_x = min((pos for _, pos in nets))
    max_x = max((pos for _, pos in nets))

    for net, pos in nets:
        if net in input_nets:
            cost += pos - min_x
        elif net in output_nets:
            cost += max_x - pos

    return cost


def _num_gate_matches(cell: Cell) -> int:
    """
    Count how many opposing gates share the same net.
    :param cell:
    :return: Number of gate matches.
    """
    sum = 0
    for a, b in zip(cell.upper, cell.lower):
        if a is not None and b is not None:
            if a.gate_net == b.gate_net:
                sum += 1
    return sum


def _cell_quality(cell: Cell, input_nets, output_nets):
    """ Quality metric for a dual stack cell.
    :param nmos:
    :param pmos:
    :param input_nets:
    :param output_nets:
    :return:
    """
    return _num_gate_matches(cell), -wiring_length_bbox(cell), \
           - _row_io_ordering_cost(cell.upper, input_nets, output_nets) \
           - _row_io_ordering_cost(cell.lower, input_nets, output_nets)


def __need_gap(left: Transistor, right: Transistor):
    """Check if a diffusion gap is required between two adjacent transistors.
    """
    if left is None or right is None:
        return False
    else:
        return left.drain_net != right.source_net


def _wiring_length_bbox1(nets: Iterable[Tuple[Hashable, float]]) -> float:
    """ Calculate 1-dimensional wiring length.
    :param nets: List[(net name, x)]
    :return: Sum of bounding box circumferences
    """

    minmax_x = dict()

    for net, x in nets:
        minmax_x[net] = (
            min(x, minmax_x.get(net, (x, x))[0]),
            max(x, minmax_x.get(net, (x, x))[1])
        )

    bbox_width = {k: mx - mn for k, (mn, mx) in minmax_x.items()}
    return sum(bbox_width.values())


def test_wiring_length_bbox1():
    nets = {('a', 0), ('b', 2), ('a', 1), ('b', 12)}
    assert _wiring_length_bbox1(nets) == 11


def _wiring_length_bbox2(nets: Iterable[Tuple[Hashable, Tuple[float, float]]]) -> float:
    """ Calculate 2-dimensional wiring length.
    :param nets: List[(net name, (x,y))]
    :return: Sum of bounding box circumferences
    """

    len_x = _wiring_length_bbox1((net, x) for net, (x, y) in nets)
    len_y = _wiring_length_bbox1((net, y) for net, (x, y) in nets)

    return len_x + len_y


def wiring_length_bbox(cell: Cell) -> float:
    """Get an indicator of the wiring length inside a cell.
    The bounding box size of the nets is used as an approximation.
    """

    net_positions = []
    for row in (cell.upper, cell.lower):
        for pos, net in enumerate(chain(*[list(t.terminals()) for t in row if t is not None])):
            net_positions.append((net, pos))

    return _wiring_length_bbox1(net_positions)


class HierarchicalPlacer(TransistorPlacer):

    def __init__(self):
        pass

    def place(self, transistors: Iterable[Transistor]) -> Cell:
        """ Place transistors by a hierarchical approach.
        The full circuit is split into sub-circuits, each containing only nmos or pmos transistors.
        The sub-circuits are placed independent of their internal placement.
        :param transistors:
        :return:
        """

        # Split into nmos / pmos
        nmos = [t for t in transistors if t.channel_type == ChannelType.NMOS]
        pmos = [t for t in transistors if t.channel_type == ChannelType.PMOS]

        nmos_graph = _transistors2graph(nmos)
        pmos_graph = _transistors2graph(pmos)

        # Partition into sub-circuits.
        nmos_graphs = partition.partition(nmos_graph)
        pmos_graphs = partition.partition(pmos_graph)

        def find_internal_placements(g: nx.MultiGraph) -> List[List[Transistor]]:
            """Find optimal-width placements for each sub-circuit.
            """
            assert len(g) > 0
            placements = _find_optimal_single_row_placements(g)
            # Append diffusion gap.
            placements = [list(chain(p, [None])) for p in placements]
            return placements

        # Find optimal-width placements for each sub-circuit.
        nmos_placements = [find_internal_placements(g) for g in nmos_graphs]
        pmos_placements = [find_internal_placements(g) for g in pmos_graphs]

        class SubCell:
            """ Abstract representation of a set of transistors.
            Only the width of the sub cell and a set of terminal names is stored.
            """

            def __init__(self, width, nets, row, id):
                """

                :param width: Width of sub cell
                :param nets: Set of net names inside the sub-cell (without power nets)
                :param row: Row index as a placement constraint. (e.g. 0 for a NMOS sub cell and 1 for a PMOS sub cell)
                :param id: Some identifier or reference to the transistors in this sub cell.
                """
                self.width = width
                self.nets = nets
                self.row = row
                self.id = id
                self.location = (None, None)

            def __repr__(self):
                return "SubCell(%d, %s)" % (self.width, self.nets)

        def subcell_from_placements(placements: List[List[Transistor]], row: int) -> SubCell:
            """ Create a SubCell from a list of possible internal placements.
            :param placements:
            :param row:
            :return:
            """
            if len(placements) == 0:
                return SubCell(0, set(), 0, placements)

            transistors = [t for t in chain(*placements) if t is not None]
            # Extract net names
            nets = {n for n in chain(*(t.terminals() for t in transistors))}
            # Strip power nets
            nets = {n for n in nets if not net_util.is_power_net(n)}

            return SubCell(len(placements[0]), nets, row, placements)

        # Get sub cells
        p_subcells = [subcell_from_placements(pl, 0) for pl in pmos_placements]
        n_subcells = [subcell_from_placements(pl, 1) for pl in nmos_placements]

        p_permuations = list(permutations(p_subcells))
        n_permuations = list(permutations(n_subcells))

        def get_subcell_net_position(subcells: Iterable[SubCell]) -> List[Tuple[Any, float]]:
            """
            Get approximate net positions given a row of sub-cells.
            Each net is treated as it would be localized in the center of the sub cell.
            :param subcells: A row of sub-cells
            :return: List[(net name, x coordinate)]
            """
            offset = 0
            nets = []
            for subcell in subcells:
                x = offset + subcell.width / 2
                nets.extend(((net, x) for net in subcell.nets))
                offset += subcell.width
            return nets

        # Get optimal placement of sub cells by trying all combinations.
        # Quality metric is the bounding box size of the nets.
        subcell_placements = product(p_permuations, n_permuations)

        # Get good sub cell placements by evaluating the bounding box sizes of the nets.
        best_subcell_placements = []
        best_wiring_length = None

        # Extract input nets.
        input_nets = net_util.get_cell_inputs(transistors)
        output_nets = set()  # TODO: extract output nets.

        for ps, ns in subcell_placements:
            ppos = get_subcell_net_position(ps)
            npos = get_subcell_net_position(ns)

            # TODO: make quality metric parametrizable.
            # Optimize for wiring length estimate and break ties with the ordering of IO nets.
            # Want input nets to be on the left.
            wiring_length = _wiring_length_bbox1(chain(ppos, npos))
            cost = wiring_length, _io_ordering_cost(chain(ppos, npos), input_nets, output_nets)

            if best_wiring_length is None or cost < best_wiring_length:
                best_subcell_placements.clear()
                best_wiring_length = cost
            if cost <= best_wiring_length:
                best_subcell_placements.append((ps, ns))

        logger.debug('Number of best sub cell placements: %d', len(best_subcell_placements))

        def get_subcell_x_offsets(subcell_row: Iterable[SubCell], offset: int = 0) -> Dict[Any, float]:
            """ Given a row of subcells find the x offset of each.
            """
            offsets = dict()
            for subcell in subcell_row:
                offsets[subcell] = offset
                offset += subcell.width
            return offsets

        def find_best_intra_subcell_placement(ps: List[SubCell], ns: List[SubCell]) -> Dict[SubCell, List[Transistor]]:
            """ Given the placement of the sub cells find the best transistor placements
            inside the subcells.
            :param ps: PMOS sub cells
            :param ns: NMOS sub cells
            :return:
            """

            # Get full cell width
            pwidth = sum((sc.width for sc in ps))
            nwidth = sum((sc.width for sc in ns))
            width = max(pwidth, nwidth)

            # Get a map of which track is occupied by which sub cell.
            height = 2
            matrix = np.ndarray((height, width), dtype=object)
            for row_idx, subcells in enumerate([ns, ps]):
                row = list(chain(*([s] * s.width for s in subcells)))
                # Pad to width
                row.extend([None] * (width - len(row)))
                matrix[row_idx, :] = row

            # Build a dependency graph: Find sub cells that depend on each others internal placement
            # (e.g. sub cells that face each other)
            dependency_graph = nx.Graph()
            # Add all nodes to graph
            for sc in chain(ns, ps):
                dependency_graph.add_node(sc)
            # Check if neighbors (in y directon) share a net. If yes, add an edge in the dependency graph.
            # Loop over pairs of neighbors in y-direction
            for a, b in zip(matrix[0:].flat, matrix[1:].flat):
                if a is not None and b is not None:
                    common_nets = a.nets & b.nets
                    if len(common_nets) > 0:
                        dependency_graph.add_edge(a, b)

            # Split dependency graph into connected sub graphs.
            subgraphs = [dependency_graph.subgraph(c) for c in nx.connected_components(dependency_graph)]
            # Construct groups of dependent cells.
            # The internal placement between different groups is independent.
            dependent_cell_groups = [list(g.nodes) for g in subgraphs]

            logger.debug('Number of sub cells: %d', len(dependent_cell_groups))

            # Find x positions of sub cells (left corner).
            x_offsets = dict()
            x_offsets.update(get_subcell_x_offsets(ps))
            x_offsets.update(get_subcell_x_offsets(ns))

            best_intra_subcell_placements = dict()

            # Iterate over groups of sub cells that must be placed together.
            for subcells in dependent_cell_groups:

                subcell_net_positions = set()
                # Add simplified net positions of all other sub cells.
                for sc in chain(ps, ns):
                    if sc not in subcells:
                        p = x_offsets[sc] + sc.width / 2
                        for net in sc.nets:
                            subcell_net_positions.add((net, p))

                min_wiring_length = None
                best_placements = []
                # Iterate over all internal placements

                for placements in product(*(s.id for s in subcells)):
                    # Get net positions for this placement
                    assert len(subcells) == len(placements)
                    net_positions = []
                    for subcell, placement in zip(subcells, placements):

                        # Expand placement into transistor terminals
                        for net_pos, net in enumerate(chain(*(t.terminals() for t in placement if t))):
                            # Ignore power nets
                            if not net_util.is_power_net(net):
                                p = x_offsets[subcell] + net_pos / 3
                                net_positions.append((net, p))

                    # Add simplified net positions of all other sub cells.
                    net_positions.extend(subcell_net_positions)

                    wiring_length = _wiring_length_bbox1(net_positions)

                    if min_wiring_length is None or wiring_length < min_wiring_length:
                        min_wiring_length = wiring_length
                        best_placements.clear()

                    if min_wiring_length == wiring_length:
                        best_placements.append(placements)

                best_placement = best_placements[0]
                assert len(subcells) == len(best_placement)
                for subcell, placement in zip(subcells, best_placement):
                    best_intra_subcell_placements[subcell] = placement

            return best_intra_subcell_placements

        # Find best internal placement for each cell in `subcells`

        best_quality = None
        best_cell = None

        for ps, ns in best_subcell_placements:
            intra_placements = find_best_intra_subcell_placement(ps, ns)

            p_row = []
            n_row = []

            for p in ps:
                transistors = intra_placements[p]
                p_row.extend(transistors)
            for n in ns:
                transistors = intra_placements[n]
                n_row.extend(transistors)

            cell = _assemble_cell(n_row, p_row)

            # Remove gaps where possible.
            cell = _remove_gaps(cell)

            input_nets = net_util.get_cell_inputs(transistors)
            output_nets = {}
            quality = _cell_quality(cell, input_nets, output_nets)

            if best_quality is None or quality > best_quality:
                best_quality = quality

            if quality == best_quality:
                best_cell = cell

        return best_cell


def _transistors2graph(transistors: Iterable[Transistor]) -> nx.MultiGraph:
    """ Create a graph representing the transistor network.
        Each edge corresponds to a transistor, each node to an electrical potential.
    """
    G = nx.MultiGraph()
    for t in transistors:
        G.add_edge(t.source_net, t.drain_net, t)
    # assert nx.is_connected(G)
    return G


def _find_optimal_single_row_placements(transistor_graph: nx.MultiGraph) -> List[List[Transistor]]:
    """ Find with-optimal single row placements of transistors.

    :param transistors: nx.MultiGraph representing the transistor network. Each edge coresponds to a transistor.
    :return: List[List[Transistor]]
    """

    even_degree_graphs = eulertours.construct_even_degree_graphs(transistor_graph)
    logger.debug('Number of even-degree graphs: %d', len(even_degree_graphs))

    all_eulertours = list(chain(*(eulertours.find_all_euler_tours(g) for g in even_degree_graphs)))

    logger.debug('Number of eulertours: %d', len(all_eulertours))
    all_eulertours = list(set((tuple(tour) for tour in all_eulertours)))
    logger.debug('Number of deduplicated eulertours: %d', len(all_eulertours))

    def cyclic_shift(l: List, shift: int) -> List:
        """ Rotate elements of a list `l` by `shift` to the left.

        :param l: List
        :param shift: Shift amount.
        :return: A shifted copy of the list.
        """
        s = islice(cycle(l), shift, len(l) + shift)
        return list(s)

    def find_optimal_shifts(tours: List[List[Transistor]]) -> List[List[Transistor]]:
        """ Find the optimal euler tours by applying a cyclic rotation to all provided euler tours.
        """
        if not tours:
            return []

        optimal = []
        optimal_len = len(tours[0])

        for tour in tours:
            tour = _trim_none(tour)
            assert len(tour) >= 1

            if tour[0].source_net != tour[-1].drain_net:
                # Can not be shifted without inserting a gap.
                tour.append(None)

            l = len(tour)
            for i in range(l):
                shifted = cyclic_shift(tour, i)
                if shifted[0] is not None:
                    shifted = _trim_none(shifted)
                    l = len(shifted)

                    if l < optimal_len:
                        optimal.clear()
                        optimal_len = l
                    if l == optimal_len:
                        optimal.append(shifted)

        # Check if there are duplicates
        _optimal = (tuple(t) for t in optimal)
        _optimal_set = set(_optimal)
        # assert len(_optimal) == len(_optimal_set)
        # Deduplicate
        optimal = list(_optimal_set)

        return optimal

    def edges2transistors(edges: Tuple[Any, Any, int]) -> List[Transistor]:
        """ Convert graph edges back to a list of transistors.
        """

        ts = []
        for l, r, t in edges:
            transistor = None
            if isinstance(t, Transistor):
                if t.source_net == l:
                    transistor = t
                else:
                    transistor = t.flipped()

                assert transistor.source_net == l and transistor.drain_net == r

            ts.append(transistor)
        return ts

    assert len(all_eulertours) > 0
    all_placements = [edges2transistors(edges) for edges in all_eulertours]
    assert len(all_placements) > 0
    all_placements = find_optimal_shifts(all_placements)

    assert len(all_placements) > 0
    return all_placements


class EulerPlacer(TransistorPlacer):

    def __init__(self):
        pass

    def place(self, transistors: Iterable[Transistor]) -> Cell:
        """Find a 2-stack placement for the tansistors.

        Transistors will be placed on a 2xn grid (2-stack cell). PMOS on the upper stack, NMOS on the lower stack.
        The placement locations are not absolute coordinates but two dimensional indices to the transistor grid.
        """
        transistors = list(transistors)
        logger.debug('Find eulerian tours.')

        nmos = [t for t in transistors if t.channel_type == ChannelType.NMOS]
        pmos = [t for t in transistors if t.channel_type == ChannelType.PMOS]

        nmos_graph = _transistors2graph(nmos)
        pmos_graph = _transistors2graph(pmos)

        all_nmos = _find_optimal_single_row_placements(nmos_graph)
        all_pmos = _find_optimal_single_row_placements(pmos_graph)

        logger.debug('Number of NMOS placements with cyclic shifts: %d', len(all_nmos))
        logger.debug('Number of PMOS placements with cyclic shifts: %d', len(all_pmos))

        if len(all_nmos) * len(all_pmos) > 100000:
            # Notify the user that the hierarchical placer might be better.
            logger.info('`EulerPlacer` will not perform well in this case. '
                        '`HierarchicalPlacer` could be a better choice.')

        # Find best nmos/pmos row pair.
        pairs = product(all_nmos, all_pmos)

        input_nets = net_util.get_cell_inputs(transistors)
        output_nets = {}

        # Assemble optimal cell candidates.
        cells = (_assemble_cell(nmos, pmos) for nmos, pmos in pairs)

        best_cells_gate_match = all_max(cells, key=_num_gate_matches)
        best_cells_wiring = all_min(best_cells_gate_match, key=wiring_length_bbox)

        def io_ordering_cost(cell: Cell):
            return _row_io_ordering_cost(cell.lower, input_nets, output_nets) + \
                   _row_io_ordering_cost(cell.upper, input_nets, output_nets)

        best_cells_io_ordering = all_min(best_cells_wiring, key=io_ordering_cost)
        best_cells = best_cells_io_ordering

        if len(best_cells) > 1:
            logger.info("Found multiple optimal placements: %d. Take the first.", len(best_cells))

        return best_cells[0]
