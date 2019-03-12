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
import klayout.db as pya
from klayout.db import Point, Shape, Polygon, Shapes, Region

from . import base as cleaner
from .base import OPolygon, SOPolygon, Orientation, OEdge

from itertools import count, product, combinations
from typing import Dict, List, Iterable, Set, Tuple, Union

import logging

from .. import tech_util

from pysmt.shortcuts import *
from pysmt.fnode import FNode
from pysmt.solvers.z3 import Z3Model, Z3Converter

import math
from ..layout.layers import *

logger = logging.getLogger(__name__)


def _polygon_to_points(poly) -> List[Tuple[int, int]]:
    """
    Convert a klayout polygon to a list of (x,y) tuples.
    The ordering of the points is reversed.
    :param poly:
    :return:
    """
    if poly is None:
        return []
    else:
        return [(p.x, p.y) for p in reversed(list(poly.each_point()))]


def _shape_to_points(shape: pya.Shape) -> List[Tuple[int, int]]:
    """ Convert a pya.Shape to a list of (x,y) tuples.
    """
    poly = shape.simple_polygon
    return _polygon_to_points(poly)


def _points_to_shape(points: Iterable[Tuple[int, int]]) -> pya.Polygon:
    """
    Create a klayout Polygon from a list of (x,y) tuples.
    :param points:
    :return:
    """
    return pya.Polygon.new([pya.Point(x, y) for x, y in points])


def clean(tech,
          shapes: Dict[str, pya.Shapes],
          white_list: Set[Tuple[str, pya.Shape]] = None,
          enable_min_area=False,
          optimize: bool = False,
          debug=False,
          solver_name: str = 'z3'):
    """ Performs DRC cleaning on the `shapes`.

    :param tech: Tech module. Defines the design rules.
    :param shapes:
    :param white_list: Set[Tuple[layer name, shape])]
        If this is used, then only the shapes in the white list will be modified.
    :param enable_min_area: Enable minimum area constraint. Disabled by default.
        Minimum area constraint showed to slow things down because they are not modelled as a integer linear program (not linear).
    :param optimize: Use z3 to find an optimal solution.
        Shape sizes are minimized.
        This will overwrite the solver name because optimizations are currently only implemented for z3.
    :param debug: Use an `UnsatCoreSolver` for debugging of unsatisfiable constraints.
        No optimizations will be performed in this mode.
    :param solver_name: Name of the SMT solver to be used.
    :return: Returns `True` iff DRC cleaning was successful.
    """

    if debug and optimize:
        logger.warning("Optimizations disabled in debug unsat core mode.")
        optimize = False

    if optimize:
        import z3

    if optimize and solver_name != 'z3':
        logger.warning("Can not use {} for optimizations. Switching to z3.".format(solver_name))
        solver_name = 'z3'

    # Graph representing the spacing between layers. Node: layer, Edge['min_spacing']: spacing rule between layers.
    spacing_graph = tech_util.spacing_graph(tech.min_spacing)

    _sympoly_map = dict()
    _counter = count()

    def get_sympoly(shape: Union[pya.Shape, pya.Polygon], layer: str) -> SOPolygon:
        """ Given a pya.Shape get the symbolic polygon.
        """

        assert isinstance(shape, pya.Shape) or isinstance(shape, pya.Polygon)

        if isinstance(shape, pya.SimplePolygon):
            poly = shape
        if isinstance(shape, pya.Polygon):
            poly = shape.to_simple_polygon()
        else:
            poly = shape.simple_polygon

        key = (layer, poly)

        if key not in _sympoly_map:
            points = _shape_to_points(shape)
            opoly = OPolygon(points=points)
            sympoly = SOPolygon('%s_%s' % (layer, next(_counter)), opoly)
            _sympoly_map[key] = sympoly

        return _sympoly_map[key]

    # Convert all polygons into symbolic polygons.
    logger.debug("Convert polygons into symbolic polygons.")

    sym_polys = {
        layer: [get_sympoly(s, layer) for s in ss.each()]
        for layer, ss in shapes.items()
    }

    # Get set of polygons that are allowed to be modified.
    if white_list is not None:
        white_list = {
            (layer, get_sympoly(shape=shape, layer=layer))
            for layer, shape in white_list
        }
    else:
        white_list = set()

    if len(white_list) == 0:
        logger.warning("No white-listed polygons. Will not perform any cleaning.")

    # Get solver instance.
    if debug:
        logger.debug("Use UnsatCoreSolver.")
        solver = UnsatCoreSolver(name=solver_name)
    else:
        solver = Solver(name=solver_name)

    # Create optimizer instances if needed.
    if optimize:
        optimizer = z3.Optimize()
        converter = solver.converter
        logger.info("SMT optimizer: z3")
    else:
        optimizer = None
        converter = None
        logger.info("SMT solver: {}".format(type(solver).__name__))

    def add_assertion(assertion: FNode, **kwargs):
        """
        Add assertion to solver and optimizer.
        :param assertion:
        :param kwargs:
        :return:
        """

        assert isinstance(assertion, FNode)
        solver.add_assertion(assertion, **kwargs)
        if optimizer:
            optimizer.add(
                converter.convert(assertion)
            )

    def minimize(objective: FNode):
        """
        Add minimization objective to optimizer.
        :param objective:
        :return:
        """
        if optimizer:
            optimizer.minimize(
                converter.convert(objective)
            )

    def maximize(objective: FNode):
        """
        Add maximization objective to optimizer.
        :param objective:
        :return:
        """
        if optimizer:
            optimizer.maximize(
                converter.convert(objective)
            )

    # WIP: detecting gate edges to preserve gates
    # def get_interacting_edges(a: SOPolygon, b: SOPolygon) -> List[OEdge]:
    #     """
    #     Find edges of `a` that interact with polygon `b`.
    #     Operates on original polygon.
    #     :param a:
    #     :param b:
    #     :return: List of OEdges that interact with b.
    #     """
    #
    #     reverse_map = dict()
    #     for orig_edge, sym_edge in zip(a.original_opolygon.edges, a.edges):
    #         (x1, y1), (x2, y2) = orig_edge.endpoints
    #         reverse_map[((x1, y1), (x2, y2))] = sym_edge
    #         reverse_map[((x2, y2), (x1, y1))] = sym_edge
    #
    #     poly_a = pya.Polygon(a.original_opolygon.points())
    #     poly_b = pya.Polygon(b.original_opolygon.points())
    #
    #     # Get edges of a that interact with b.
    #     interacting = pya.Edges(poly_a).interacting(pya.Region(poly_b))
    #
    #     # Convert back to corresponding symbolic edges.
    #     interacting_symbolic_edges = [reverse_map[((e.x1, e.y1), (e.x2, e.y2))]
    #                                   for e in interacting
    #                                   ]
    #     return interacting_symbolic_edges
    #
    # for active_shape in shapes[l_active].each():
    #     # Find touching poly-silicon shapes
    #     active_region = pya.Shapes()
    #     active_region.insert(active_shape)
    #     interacting = pya.Region(shapes[l_poly]).interacting(pya.Region(active_region))
    #     for i in interacting:
    #         gate_edges = get_interacting_edges(get_sympoly(i, l_poly),
    #                                            get_sympoly(active_shape, l_active))

    # Add assertions

    # Fix all polygons that are not in the white list.
    fix_counter = 0
    for layer, polys in sym_polys.items():
        for poly in polys:
            if (layer, poly) not in white_list:
                add_assertion(poly.assert_absolute_fixed(),
                              named='fixed shapes ({})'.format(layer))
                fix_counter += 1

    logger.debug("Fixed polygons: {}".format(fix_counter))
    logger.debug("Free polygons: {}".format(len(white_list)))

    # Fix via shapes but allow them to be moved.
    logger.debug("Assuming immutable via shapes.")
    via_layers = set(tech.via_layers.values())
    logger.debug("Add constraint for relative immutable shapes: {}".format(via_layers))
    relative_fixed_layers = via_layers
    for l in relative_fixed_layers:
        for poly in sym_polys[l]:
            add_assertion(cleaner.preserve_shape(poly),
                          named='relative immutable shapes ({})'.format(l))

    # Allow free polygons to be changed in shape but the underlying structure must be preserved.
    fixed_form_layers = shapes.keys()
    for l in fixed_form_layers:
        for poly in sym_polys[l]:
            add_assertion(poly.assert_preserved_edge_orientation(),
                          named='fixed edge orientation ({})'.format(l))

    # TODO: more generic, based on technology file
    containtement_constraints = [
        ([l_abutment_box], [l_active, l_diff_contact, l_poly_contact, l_via1, l_poly, l_metal1, l_metal2]),
        ([l_active], [l_diff_contact]),
        ([l_poly], [l_poly_contact]),
        ([l_metal1], [l_poly_contact, l_diff_contact, l_via1]),
        ([l_metal2], [l_via1]),
    ]

    def assert_inside_naive(outer_poly: SOPolygon, inner_poly: SOPolygon, min_inside=1) -> FNode:
        """ Create a constraint that asserts that `inner_poly` is contained in `outer_poly`.
        Requirement: The original polygons are required to fulfill this property already!

        :param outer_poly: Container polygon.
        :param inner_poly: Inner polygon.
        :param min_inside: Minimum distance of the inner polygon to the container.
        :return: FNode
        """

        v1 = outer_poly.verticals()
        v2 = inner_poly.verticals()

        h1 = outer_poly.horizontals()
        h2 = inner_poly.horizontals()

        constraints = []

        for v in v1:
            for u in v2:
                if v.attrs['orig_offset'] < u.attrs['orig_offset']:
                    constraints.append(u.offset - v.offset >= min_inside)
                elif v.attrs['orig_offset'] > u.attrs['orig_offset']:
                    constraints.append(v.offset - u.offset >= min_inside)

        for h in h1:
            for i in h2:
                if h.attrs['orig_offset'] < i.attrs['orig_offset']:
                    constraints.append(i.offset - h.offset >= min_inside)
                elif h.attrs['orig_offset'] > i.attrs['orig_offset']:
                    constraints.append(h.offset - i.offset >= min_inside)

        return And(*constraints)

    min_enclosure = dict()

    # Add half-spacing rules.
    # Half-spacing rules assert that spacing rules are hold when abutting two cells.
    # The approach is to enforce a spacing to the cell boundary.
    logger.debug("Add half-spacing constraints.")
    for l1, l2, data in spacing_graph.edges(data=True):

        half_spacing = (data['min_spacing'] + 1) // 2

        outer = l_abutment_box
        for inner in [l1, l2]:
            min_enc = min_enclosure.get((outer, inner), 0)
            min_enclosure[(outer, inner)] = max(min_enc, half_spacing)

    # Add minimum enclosure rules (of vias).
    for (l1, l2), via_layer in tech.via_layers.items():
        for outer in (l1, l2):
            min_enclosure[(outer, via_layer)] = tech.minimum_enclosure[(outer, via_layer)]

    for outer_layers, inner_layers in containtement_constraints:
        for outer_layer, inner_layer in product(outer_layers, inner_layers):

            min_inside = min_enclosure.get((outer_layer, inner_layer), 1)

            for outer_shape in shapes[outer_layer].each():
                # logger.debug("outer_shape = %s", outer_shape)
                _outer = Shapes()
                _outer.insert(outer_shape)
                _outer = Region(_outer)
                inner = Region(shapes[inner_layer])

                inside = inner.inside(_outer)
                outer_poly = get_sympoly(outer_shape, outer_layer)
                for inner_shape in inside.each():
                    inner_poly = get_sympoly(inner_shape, inner_layer)

                    if (outer_layer, outer_poly) in white_list or (inner_layer, inner_poly) in white_list:
                        # Add constraint only if both polygons are modifiable.
                        add_assertion(
                            assert_inside_naive(outer_poly, inner_poly, min_inside=min_inside),
                            named="min inside {}-{}".format(outer_layer, inner_layer)
                        )

    # Min width
    logger.debug("Add minimum width constraints.")
    for layer, min_width in tech.minimum_width.items():
        for p in sym_polys[layer]:
            add_assertion(
                cleaner.min_width_of_polygons([p], min_width=min_width),
                named="min width ({})".format(layer)
            )

    # Min notch
    logger.debug("Add minimum notch constraints.")
    for layer, min_notch in tech.minimum_notch.items():
        for p in sym_polys[layer]:
            add_assertion(
                cleaner.min_notch_of_polygons([p], min_notch=min_notch),
                named="min notch ({})".format(layer)
            )

    # Min area
    if enable_min_area:
        logger.debug("Add minimum area constraints.")
        # logger.warning('Enabled min_area constraints. (Results in non-linear problem and can slow things down.)')
        for layer, min_area in tech.min_area.items():
            for p in sym_polys[layer]:
                if len(p.points()) == 4 and (layer, p) in white_list:
                    if min_area > 0:
                        # add_assertion(
                        #     p.area() >= min_area,
                        #     named="min area {}".format(layer)
                        # )

                        # Create a square shape.
                        side = int(math.sqrt(min_area) + 1)
                        for e in p.edges:
                            add_assertion(
                                e.length() >= side if optimize else Equals(e.length(), Int(side)),
                                named="min area {}".format(layer)
                            )

    # Min spacing
    logger.debug('Add minimum spacing rules.')
    # TODO:
    logger.warning("Fixed shapes are excluded from spacing rules.")
    for (layer1, layer2), spacing in tech.min_spacing.items():

        min_dist_v = spacing
        min_dist_h = min_dist_v

        if layer1 == layer2:
            # Intra-layer spacing
            pairs = combinations(sym_polys[layer1], 2)
        else:
            # Inter-layer spacing
            pairs = product(sym_polys[layer1], sym_polys[layer2])

        for p1, p2 in pairs:

            # Don't add constraints if both polygons are immutable.
            if (layer1, p1) not in white_list and (layer2, p2) not in white_list:
                # TODO: include fixed layers? (implicit DRC check)
                continue

            for e1, e2 in product(p1.verticals(), p2.verticals()):
                add_assertion(
                    Implies(cleaner.edge_projections_overlap(e1, e2, margin=min_dist_h),
                            cleaner.preserve_absolute_edge_order([e1, e2], min_distance=min_dist_v)),
                    named='min spacing horizontal {}-{}'.format(layer1, layer2)
                )

            for e1, e2 in product(p1.horizontals(), p2.horizontals()):
                add_assertion(
                    Implies(cleaner.edge_projections_overlap(e1, e2, margin=min_dist_v),
                            cleaner.preserve_absolute_edge_order([e1, e2], min_distance=min_dist_h)),
                    named='min spacing vertical {}-{}'.format(layer1, layer2)
                )

    if optimizer:
        # Optimization objectives
        for layer, polys in sym_polys.items():
            for poly in polys:
                for edge in poly.edges:
                    # TODO: add constraints sorted by original edge length descending.
                    # -> better area optimization because objectives are priorized lexicographically.
                    if edge.orientation in [Orientation.UP, Orientation.LEFT]:
                        # Upper or right edge.
                        minimize(edge.offset)
                    else:
                        # Lower or left edge.
                        maximize(edge.offset)

        logger.info('Invoke SMT optimizer')
        sat = optimizer.check()
        logger.info('SMT optimize result: %s', 'SAT' if sat else 'UNSAT')
    else:
        logger.info('Invoke SMT solver')
        sat = solver.check_sat()
        logger.info('SMT result: %s', 'SAT' if sat else 'UNSAT')

    if sat:
        if optimizer:
            model = Z3Model(solver.environment, optimizer.model())
        else:
            model = solver.get_model()

        # Convert back to constant polygons.
        solutions = {
            l: [p.to_opolygon(model) for p in polys]
            for l, polys in sym_polys.items()
        }

        new_shapes = {
            l: [_points_to_shape(poly.points()) for poly in polys]
            for l, polys in solutions.items()
        }

        # Update shapes with solutions
        for l, new_s in new_shapes.items():
            s = shapes[l]
            s.clear()
            s.insert(pya.Region(new_s))

        return True

    else:
        logger.error('UNSAT: Constraints not satisfiable.')
        if debug:
            core = solver.get_named_unsat_core()
            logger.info('unsat core: %s', list(core.keys()))
            logger.debug('unsat core: %s', core)
        return False
