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
import sys

import networkx as nx
from typing import Tuple, Iterable

# klayout.db should not be imported if script is run from KLayout GUI.
if 'pya' not in sys.modules:
    import klayout.db as pya


def is_inside(p: Tuple[int, int], r: pya.Region, margin: int):
    """ Test if point is inside region with some margin.
    """
    p = pya.Point(*p)
    v = pya.Vector(margin, margin)
    box = pya.Region(pya.Box(p - v, p + v))
    return not box.inside(r).is_empty()


def interacts(p: Tuple[int, int], r: pya.Region, margin: int):
    """ Test if point is close (margin) to region.
    """
    p = pya.Point(*p)
    v = pya.Vector(margin, margin)
    box = pya.Region(pya.Box(p - v, p + v))
    return not r.interacting(box).is_empty()


def is_edge_inside(graph_edge, r: pya.Region, margin: int):
    """Checks if the edge of the graph lies inside the region when mapped to a layout.
    """

    assert margin > 0, "Path with width 0 is considered non-existent."

    (l1, (x1, y1)), (l2, (x2, y2)) = graph_edge

    p = pya.Region(pya.Path([pya.Point(x1, y1), pya.Point(x2, y2)], margin, margin, margin))

    return not p.inside(r).is_empty()


def inside(points: Iterable[Tuple[int, int]], r: pya.Region, margin: int):
    """ Get all points that lie inside the region.
    """
    return [p for p in points if is_inside(p, r, margin)]


def interacting(points: Iterable[Tuple[int, int]], r: pya.Region, margin: int):
    """ Get all points that are close (margin) to the region.
    """
    return [p for p in points if interacts(p, r, margin)]


def edges_inside(g: nx.Graph, r: pya.Region, margin: int):
    """ Get all graph edges that lie inside the region.
    """
    return [e for e in g.edges if is_edge_inside(e, r, margin)]
