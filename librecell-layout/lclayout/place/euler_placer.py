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


def _assemble_cell(nmos: List[Transistor], pmos: List[Transistor]) -> Cell:
    """ Build a Cell object from a nmos and pmos row.
    :param nmos:
    :param pmos:
    :return:
    """
    width = max(len(nmos), len(pmos))
    cell = Cell(width)
    for i, t in enumerate(pmos):
        cell.upper[i] = t

    for i, t in enumerate(nmos):
        cell.lower[i] = t
    return cell


def _io_ordering_cost(row: List[Transistor], input_nets: Set[Any], output_nets: Set[Any]) -> float:
    """ Return a low cost if input nets are placed on the left side and outputs on the right side.
    """

    w = len(row)
    cost = 0
    for pos, net in enumerate(chain(*[[t.left, t.gate, t.right] for t in row if t is not None])):
        if net in input_nets:
            cost += pos
        elif net in output_nets:
            cost += w - pos
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
            if a.gate == b.gate:
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
    return _num_gate_matches(cell), -wiring_length_bbox(cell), - _io_ordering_cost(cell.upper, input_nets,
                                                                                   output_nets) - \
           _io_ordering_cost(cell.lower, input_nets, output_nets)


def __need_gap(left: Transistor, right: Transistor):
    """Check if a diffusion gap is required between two adjacent transistors.
    """
    if left is None or right is None:
        return False
    else:
        return left.right != right.left


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

    # TODO use _wiring_length_bbox1
    # Find for each net its leftmost (min) and rightmost (max) occurrence.
    def find_all_bbox_x(row, minmax_x):
        for pos, net in enumerate(chain(*[[t.left, t.gate, t.right] for t in row if t is not None])):
            minmax_x[net] = (
                min(pos, minmax_x.get(net, (pos, pos))[0]),
                max(pos, minmax_x.get(net, (pos, pos))[1])
            )

    net_positions = []
    for row in (cell.upper, cell.lower):
        for pos, net in enumerate(chain(*[[t.left, t.gate, t.right] for t in row if t is not None])):
            net_positions.append((net, pos))

    return _wiring_length_bbox1(net_positions)


class HierarchicalPlacer(TransistorPlacer):

    def __init__(self):
        pass

    def place(self, transistors: Iterable[Transistor]) -> Cell:
        """ Place transistors by a hierarchical approach.
        The full circuit is split into subcircuits, each containing only nmos or pmos transistors.
        The subcircuits are placed independent of their internal placement.
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
                :param nets: Set of net names inside the subcell (without power nets)
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
            Get approximate net positions given a row of subcells.
            Each net is treated as it would be localized in the center of the sub cell.
            :param subcells: A row of subcells
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

        for ps, ns in subcell_placements:
            ppos = get_subcell_net_position(ps)
            npos = get_subcell_net_position(ns)

            # TODO: make quality metric more flexible.
            wiring_length = _wiring_length_bbox1(chain(ppos, npos))

            if best_wiring_length is None or wiring_length < best_wiring_length:
                best_subcell_placements.clear()
                best_wiring_length = wiring_length
            if wiring_length <= best_wiring_length:
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

            # TODO: remove gaps where possible

            # Remove Nones at begin and end of row
            p_row = _trim_none(p_row)
            n_row = _trim_none(n_row)

            cell = _assemble_cell(n_row, p_row)

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
        G.add_edge(t.left, t.right, t)
    assert nx.is_connected(G)
    return G


def _find_optimal_single_row_placements(transistor_graph: nx.MultiGraph) -> List[List[Transistor]]:
    """ Find with-optimal single row placements of transistors.

    :param transistors: nx.MultiGraph representing the transistor network. Each edge corresponts to a transistor.
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

            if tour[0].left != tour[-1].right:
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
                if t.left == l:
                    transistor = t
                else:
                    transistor = t.flipped()

                assert transistor.left == l and transistor.right == r

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

        # Find best nmos/pmos row pair.

        pairs = product(all_nmos, all_pmos)

        input_nets = net_util.get_cell_inputs(transistors)
        output_nets = {}

        cells = (_assemble_cell(nmos, pmos) for nmos, pmos in pairs)

        best_cells_gate_match = all_max(cells, key=_num_gate_matches)
        best_cells_wiring = all_min(best_cells_gate_match, key=wiring_length_bbox)

        def io_ordering_cost(cell: Cell):
            return _io_ordering_cost(cell.lower, input_nets, output_nets) + \
                   _io_ordering_cost(cell.upper, input_nets, output_nets)

        best_cells_io_ordering = all_min(best_cells_wiring, key=io_ordering_cost)
        best_cells = best_cells_io_ordering

        if len(best_cells) > 1:
            logger.info("Found multiple optimal placements: %d. Take the first.", len(best_cells))

        return best_cells[0]
