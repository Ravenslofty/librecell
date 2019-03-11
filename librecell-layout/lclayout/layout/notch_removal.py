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

