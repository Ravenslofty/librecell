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
