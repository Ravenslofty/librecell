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
from pysmt.shortcuts import *
from pysmt.fnode import FNode
from pysmt.environment import get_env

from copy import deepcopy
from itertools import tee, permutations, chain, count, islice, cycle, combinations
from typing import Iterable, List, Tuple
from enum import Enum

get_env().enable_infix_notation = True


def window(iterable: Iterable, size: int) -> Iterable[Tuple]:
    """
    Get a sliding window iterator.
    :param iterable:
    :param size: Window size.
    :return: Iterator returning sliding windows.
    """
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


def argmin(l: Iterable, key=lambda x: x) -> int:
    """
    Get index (position) of the minimal element.
    :param l: Iterable containing the elements.
    :param key: A lambda function to be applyied to the elements before.
    :return: Index to the minimal element.
    """
    m, i = min(zip(map(key, l), count()))
    return i


class Orientation(Enum):
    """ Represents the orientation of a rectilinear polygon edge.
    """

    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    def is_horizontal(self):
        """
        Returns `True` iff the edge points either to the right or to the left.
        :return:
        """
        return self == Orientation.RIGHT or self == Orientation.LEFT

    def is_vertical(self):
        """
        Returns `True` iff the edge points either downwards or upwards.
        :return:
        """
        return self == Orientation.UP or self == Orientation.DOWN

    def angle(self, other):
        """ Return the angle between `other` and `self` in degrees.

        Returns: 0, 90, 180 or 270
        """
        return ((4 + other.value - self.value) % 4) * 90

    def __str__(self):
        arrow_char = {
            self.RIGHT: '→',
            self.UP: '↑',
            self.LEFT: '←',
            self.DOWN: '↓'
        }
        return arrow_char[self]


class OEdge:
    """ Representation of a rectilinear line.

    Parameters:
    offset: Distance to (0, 0): x offset for vertical lines, y offset for horizontal lines.
    orientation: 'r', 'u', 'l' or 'd'
    original_coords: Original start and end points.
    endpoints: Start and end point.
    interval: The interval spanned by the edge.
    """

    def __init__(self, offset, orientation: Orientation,
                 original_coords: Tuple[Tuple[int, int], Tuple[int, int]]):
        self.offset = offset  # Distance to (0, 0)
        self.orientation = orientation  # Orientaion of the edge.
        self.original_coords = original_coords
        self.endpoints = None
        self.interval = None  # Sorted endpoints such that start < end.
        self.attrs = dict()

    def from_points(start: Tuple[int, int], end: Tuple[int, int]):
        """ Create an edge from start and end point.
        """
        (x1, y1), (x2, y2) = start, end
        assert (x1 == x2) ^ (y1 == y2), Exception("Edge must be either horizontal or vertical.")

        if x1 == x2:
            offset = x1
            if y1 < y2:
                orientation = Orientation.UP
            else:
                orientation = Orientation.DOWN
        else:
            offset = y1
            if x1 < x2:
                orientation = Orientation.RIGHT
            else:
                orientation = Orientation.LEFT

        return OEdge(offset, orientation, (start, end))

    def length(self):
        """
        Calculates the length of the edge. This is only possible if the `interval` is set.
        :return:
        """
        assert self.interval is not None, Exception("Cannot calculate edge length. (`interval` not set)")
        return self.interval[1] - self.interval[0]

    def angle(self, other) -> int:
        """
        Calculate the relative angle to another edge.
        :param other:
        :return:
        """
        return self.orientation.angle(other.orientation)

    def is_horizontal(self) -> bool:
        return self.orientation.is_horizontal()

    def is_vertical(self) -> bool:
        return self.orientation.is_vertical()

    def __repr__(self):
        return "%s %s" % (self.offset, self.orientation)


def points2opolygon(points: List[Tuple[int, int]]) -> List[OEdge]:
    """ Convert an orthogonal polygon into a unique, overhead less representation.

    Returns
    -------
    Returns a list of OEdge. The list contains horizontal and vertical edges in a strictly alternating order.
    """
    n = len(points)

    if n == 0:
        return []

    assert n % 2 == 0, Exception("Cannot convert an odd number of points into a orthogonal polygon.")

    # Find index of smallest point sorted lexicographically by (y,x).
    lower_left_idx = argmin(points, key=lambda p: tuple(reversed(p)))
    prev_point = points[lower_left_idx]

    e1 = OEdge.from_points(prev_point, points[(lower_left_idx + 1) % n])
    assert e1.orientation == Orientation.RIGHT, Exception("Polgon is not oriented CCW.")

    edges = []
    horiz = True
    for i in range(n):
        idx = (i + lower_left_idx + 1) % n
        x1, y1 = prev_point
        x2, y2 = points[idx]

        e = prev_point, points[idx]

        assert x1 == x2 or y1 == y2, Exception("Edge is not orthogonal.")

        edge = OEdge.from_points(*e)

        if horiz:
            assert y1 == y2
            assert edge.is_horizontal()
        else:
            assert x1 == x2
            assert edge.is_vertical()

        edges.append(edge)
        horiz = not horiz
        prev_point = x2, y2

    return edges


class OPolygon:
    """ Polygon with orthogonal edges.
    Edges and points are defined counter clock-wise.
    """

    def __init__(self, points=None, edges=None):
        assert points is not None or edges is not None, Exception("Either points or edges must be given.")

        self.edges = None
        if points is not None:
            self.edges = points2opolygon(points)
        elif edges is not None:
            self.edges = edges

    def __repr__(self):
        return "OPolygon(%s)" % self.edges

    def horizontals(self) -> List[OEdge]:
        """
        :return: Returns a list of all horizontal edges.
        """
        return [e for e in self.edges if e.is_horizontal()]

    def verticals(self) -> List[OEdge]:
        """
        :return: Returns a list of all vertical edges.
        """
        return [e for e in self.edges if e.is_vertical()]

    def edges_by_orientation(self, orientation: Orientation) -> List[OEdge]:
        """
        Get a list of edges filtered by orientation.
        :param orientation:
        :return:
        """
        return [e for e in self.edges if e.orientation == orientation]

    def points(self) -> List:
        """ Get points representing the polygon.
        Returns
        -------
        [(x1, y1), ...]
        """
        return opolygon2points(self.edges)


def opolygon2points(opolygon: OPolygon) -> List:
    """ Get the points representing the opolygon.
    """
    n = len(opolygon)
    if n == 0:
        return []

    assert n >= 4, Exception("Not a valid orthogonal polygon.")
    assert n % 2 == 0, Exception("Not a valid orthogonal polygon.")
    horiz = True
    points = []
    for j in range(n):
        i = j - 1

        if horiz:
            p = opolygon[i].offset, opolygon[j].offset
            points.append(p)
        else:
            p = opolygon[j].offset, opolygon[i].offset
            points.append(p)

        horiz = not horiz

    return points


class SOPolygon:
    """ Symbolic orthogonal polygon.

    Similar to OPolygon, but edge offsets are now pySMT symbols.
    """

    def __init__(self, name: str, opolygon: OPolygon):
        self.original_opolygon = opolygon
        self.edges = deepcopy(opolygon.edges)

        # Replace edge offsets by pySMT symbols.
        for i, e in enumerate(self.edges):
            e.attrs['orig_offset'] = e.offset
            if 'fixed' in e.attrs:
                # TODO remove this
                e.offset = Int(e.offset)
            else:
                e.offset = Symbol("%s_%s" % (name, i), INT)

        # Add endpoint pointers to edges.
        n = len(self.edges)
        for j in range(n):
            i = j - 1
            k = (j + 1) % n

            # Get previous, current and next edge
            e_prev = self.edges[i]
            e = self.edges[j]
            e_next = self.edges[k]

            e.endpoints = (e_prev.offset, e_next.offset)

            # Make sure e.interval is sorted ascending.
            if e.orientation in [Orientation.RIGHT, Orientation.UP]:
                e.interval = (e_prev.offset, e_next.offset)
            else:
                e.interval = (e_next.offset, e_prev.offset)

        self.name = name

    def __repr__(self):
        return "SOPolygon(%s)" % self.edges

    def horizontals(self) -> List[OEdge]:
        return [e for e in self.edges if e.is_horizontal()]

    def verticals(self) -> List[OEdge]:
        return [e for e in self.edges if e.is_vertical()]

    def edges_by_orientation(self, orientation) -> List[OEdge]:
        return [e for e in self.edges if e.orientation == orientation]

    def points(self) -> List[Tuple[FNode, FNode]]:
        return opolygon2points(self.edges)

    def area(self) -> FNode:
        """
        Get a formula for the area of the symbolic polygon.
        Warning: This returns a non-linear formula which can lead to very slow solving.
        Some solvers will not even support this.
        :return: Formula for the area.
        """
        vs = self.verticals()

        summands = []
        for v in vs:
            start, end = v.endpoints
            a = v.offset * (end - start)
            summands.append(a)

        return sum(summands)

    def assert_preserved_edge_orientation(self) -> FNode:
        """ Assert that an edge keeps its original direction.

        E.g. an edge originally pointing to the right is not allowed to point to the left.

        Returns
        -------
        pysmt.fnode.FNode
        """
        return And(*[e.interval[0] <= e.interval[1] for e in self.edges])

    def assert_absolute_fixed(self) -> FNode:
        """ Get a constraint that fixes the polygon to its original shape and absolute location.
        """
        return And(*[Equals(e.offset, Int(e.attrs['orig_offset'])) for e in self.edges])

    def to_opolygon(self, model):
        """ Convert the symbolic polygon back to a constant polygon.

        Parameters
        ----------
        model: The solution of the SMT solver.
        """
        const_edges = []
        for e in self.edges:
            off = e.offset
            off2 = model.get_py_value(off) if off.is_constant() else model[off].constant_value()
            edge = OEdge(off2, e.orientation, e.original_coords)

            const_edges.append(edge)

        return OPolygon(edges=const_edges)


def preserve_absolute_edge_order(edges: List[OEdge], min_distance: int = 0) -> FNode:
    """ Get a pySMT formula which constrains the edges to preserve their order.
    """

    if len(edges) > 0:
        for e in edges:
            assert e.orientation.is_vertical() == edges[0].orientation.is_vertical(), Exception(
                "Cannot mix edge orientations.")

    # Sort by original offset
    s = sorted(edges, key=lambda e: e.attrs['orig_offset'])

    constraints = [(b.offset - a.offset) >= min_distance
                   for a, b in window(s, 2)
                   if a.offset != b.offset
                   ]

    return And(*constraints)


def edges_contain_polygon(a: OEdge, b: OEdge) -> bool:
    """
    Check if the inside of the polygon is between the edges.
    :param a:
    :param b:
    :return:
    """
    return a.offset < b.offset and a.orientation == Orientation.DOWN and b.orientation == Orientation.UP \
           or a.offset < b.offset and a.orientation == Orientation.RIGHT and b.orientation == Orientation.LEFT


def min_width(edges: List[OEdge],
              min_distance: int,
              min_distance_other_direction: int = None,
              notch_instead_width=False) -> FNode:
    """ Get a pySMT formula which constrains the shape to a minimum width.
    :param edges: List of either all horizontal or vertical edges of a polygon.
    """

    if min_distance_other_direction is None:
        min_distance_other_direction = min_distance

    if len(edges) > 0:
        for e in edges:
            assert e.orientation.is_vertical() == edges[0].orientation.is_vertical(), Exception(
                "Cannot mix edge orientations.")

    # Sort by original offset
    s = sorted(edges, key=lambda e: e.attrs['orig_offset'])

    def is_edge_pair_relevant(a: OEdge, b: OEdge):
        if notch_instead_width:
            return edges_contain_polygon(b, a)
        else:
            return edges_contain_polygon(a, b)

    # Create conditional constraints:
    # If the edge projections overlap, then the minimum width must be asserted.
    # Edges of length 0 can be ignored.
    # If `notch_instead_width` is set, then invert the selection of edges.
    constraints = [
        Implies(
            And(
                a.length() > 0,
                b.length() > 0,
                edge_projections_overlap(a, b, margin=min_distance_other_direction)
            ),
            (b.offset - a.offset) >= min_distance
        )
        for a, b in combinations(s, 2)
        if a.offset != b.offset and is_edge_pair_relevant(a, b)
    ]

    return And(*constraints)


def min_width_poly(poly: SOPolygon, min_distance: int) -> FNode:
    """
    Get a minimum width constraint for a symbolic polygon.
    :param polys:
    :param min_distance:
    :return:
    """
    return And(
        min_width(poly.horizontals(), min_distance=min_distance),
        min_width(poly.verticals(), min_distance=min_distance)
    )


def min_width_of_polygons(polys: Iterable[SOPolygon], min_width: int) -> FNode:
    """ Create a minimum width constraint for all polygons.
    :param polys:
    :param min_width:
    :return: Returns constraint formula.
    """
    constraints = (min_width_poly(p, min_width) for p in polys)
    return And(*constraints)


def min_notch(edges: List[OEdge],
              min_distance: int,
              min_distance_other_direction: int = None) -> FNode:
    """ Get a pySMT formula which constrains the shape to a minimum notch width.
    """

    return min_width(edges, min_distance=min_distance, min_distance_other_direction=min_distance_other_direction,
                     notch_instead_width=True)


def min_notch_poly(poly: SOPolygon, min_notch: int) -> FNode:
    """
    Get a minimum notch constraint for a symbolic polygon.
    :param polys:
    :param min_notch:
    :return:
    """
    return And(
        min_width(poly.horizontals(), min_distance=min_notch, notch_instead_width=True),
        min_width(poly.verticals(), min_distance=min_notch, notch_instead_width=True)
    )


def min_notch_of_polygons(polys: Iterable[SOPolygon], min_notch: int) -> FNode:
    """ Create a minimum notch constraint for all polygons.
    :param polys:
    :param min_notch:
    :return: Returns constraint formula.
    """
    constraints = (min_notch_poly(p, min_notch) for p in polys)
    return And(*constraints)


def preserve_absolute_edge_order_of_polygons(polys: Iterable[SOPolygon], min_distance: int = 0) -> FNode:
    """ Create a constraint that asserts the ordering of all vertical edges in x direction and all horizontal
    edges in y direction.
    :param polys:
    :param min_distance:
    :return:
    """
    # Get all edges
    polys = list(polys)
    horizontals = list(chain(*[p.horizontals() for p in polys]))
    verticals = list(chain(*[p.verticals() for p in polys]))

    return And(
        preserve_absolute_edge_order(horizontals, min_distance=min_distance),
        preserve_absolute_edge_order(verticals, min_distance=min_distance)
    )


def edge_projections_overlap(e1: OEdge, e2: OEdge, margin: int = 0) -> FNode:
    """
    Check if two edges of the same orientation over lap when projected onto the axis with the same orientation.
    :param e1:
    :param e2:
    :return:
    """

    a1, a2 = e1.interval
    b1, b2 = e2.interval
    if margin > 0:
        a1 = a1 - margin
        a2 = a2 + margin

    no_overlap = Or(
        (a1 < b1) & (a1 < b2) & (a2 < b1) & (a2 < b2),
        (b1 < a1) & (b1 < a2) & (b2 < a1) & (b2 < a2),
    )
    has_overlap = Not(no_overlap)
    return has_overlap


def preserve_relative_edge_distance(edges: List[OEdge]) -> FNode:
    """ Get a pySMT formula which constrains the edges to preserve their order.
    """

    if len(edges) > 0:
        for e in edges:
            assert e.orientation.is_vertical() == edges[0].orientation.is_vertical(), Exception(
                "Cannot mix edge orientations.")

    # Sort by original offset
    s = sorted(edges, key=lambda e: e.attrs['orig_offset'])

    constraints = [
        (b.offset - a.offset).Equals(b.attrs['orig_offset'] - a.attrs['orig_offset'])
        for a, b in window(s, 2)
    ]

    return And(*constraints)


def preserve_shape(poly: SOPolygon) -> FNode:
    """ Get a constraint that fixes the polygon shape to its original shape.
    """
    return And(
        preserve_relative_edge_distance(poly.verticals()),
        preserve_relative_edge_distance(poly.horizontals())
    )


'''     
def minimal_unconditional_edge_distance(edges, min_distance):
        """ Get a pySMT formula which constrains the edges to have a given distance.
        """
        if len(edges) > 0:
                for e in edges:
                        assert e.orientation == edges[0].orientation, Exception("Cannot mix edge orientations.")

        constraints = [ Pow((a.offset - b.offset),2) >= min_distance**2 for a,b in permutations(edges, 2)]
                
        return And(*constraints)        
'''


def test_drc_cleaner():
    points = [(1, 0), (1, 1), (0, 1), (0, 0)]
    op = OPolygon(points)
    print(op)

    points2 = [(11, 0), (11, 1), (10, 1), (10, 0)]
    op2 = OPolygon(points)

    points2 = op.points()
    print(points2)
    print(op.verticals())

    op.verticals()[0].attrs['fixed'] = True
    sop = SOPolygon('p1', op)
    print(sop)

    sop2 = SOPolygon('p2', op2)

    s = Solver()
    s.add_assertion(sop.assert_preserved_edge_orientation())
    s.add_assertion(preserve_absolute_edge_order(sop.horizontals(), min_distance=1))
    s.add_assertion(preserve_absolute_edge_order(sop.verticals(), min_distance=1))
    s.add_assertion(preserve_relative_edge_distance(sop.verticals()))

    s.add_assertion(sop2.assert_preserved_edge_orientation())
    s.add_assertion(preserve_absolute_edge_order(sop2.horizontals(), min_distance=1))
    s.add_assertion(preserve_absolute_edge_order(sop2.verticals(), min_distance=1))

    area = Symbol('area', INT)
    s.add_assertion(Equals(area, sop.area()))

    print(s.assertions)
    sat = s.check_sat()

    print('sat = ', sat)
    if sat:
        model = s.get_model()

        solved_op = sop.to_opolygon(model)
        print(model)
        print(solved_op)
        print(solved_op.points())
        print()
    else:
        pass
        # ucore = s.get_unsat_core()
        # print(ucore)
