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
import klayout.db as pya


def fill_notches(region: pya.Region, minimum_notch: int) -> pya.Region:
    """ Fill notches in a pya.Region.
    :param region:
    :param minimum_notch:
    :return:
    """

    notches = region.notch_check(minimum_notch)
    spaces = region.space_check(minimum_notch)
    notches = list(notches) + list(spaces)
    s = pya.Shapes()
    s.insert(region)
    for edge_pair in notches:
        a, b = edge_pair.first, edge_pair.second
        # Find smaller edge (a)
        a, b = sorted((a, b), key=lambda e: e.length())

        # Construct a minimal box to fill the notch
        box = a.bbox()
        # Extend box of shorted edge by points of longer edge
        box1 = box + b.p1
        box2 = box + b.p2

        # Take the smaller box.
        min_box = min([box1, box2], key=lambda b: b.area())

        s.insert(min_box)

    result = pya.Region(s)
    result.merge()
    return result

