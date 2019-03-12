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
from ..place.place import Transistor, ChannelType

from .grid_helpers import *
from .layers import *
from typing import Any, Dict, Tuple
import sys

# klayout.db should not be imported if script is run from KLayout GUI.
if 'pya' not in sys.modules:
    import klayout.db as pya


class TransistorLayout:
    """ Layout representation of transistor.
    
    Contains the shapes of the transistor plus shapes that mark possible locations of contacts.
    """

    def __init__(self, gate, active, source_box, drain_box, terminals, nwell=None):
        """

        :param gate: pya.Path
          Layout of PC gate.
        
        :param active: pya.Box
          Layout of l_active.
        
        :param source_box: pya.Box
          Marks possible locations for contacts to source.
        
        :param drain_box: pya.Box
          Marks possible locations for contacts to drain.
          
        """
        # Transistor polygons/paths
        self.gate = gate
        self.active = active

        # n-well for P-mos
        self.nwell = nwell

        # Terminal areas for source and drain.
        self.source_box = source_box
        self.drain_box = drain_box

        # Terminal nodes for gate.
        self.terminals = terminals


def draw_transistor(t: TransistorLayout, shapes: Dict[Any, pya.Region]):
    """ Draw a TransistorLayout.

    :param t: TransistorLayout
    
    :param shapes: Dict[layer name, pya.Shapes]
      A dict mapping layer names to pya.Shapes.
    """
    shapes[l_active].insert(t.active)

    if t.nwell:
        # For PMOS only.
        shapes[l_nwell].insert(t.nwell)

    shapes[l_poly].insert(t.gate)


def create_transistor_layout(t: Transistor, loc: Tuple[int, int], tech) -> TransistorLayout:
    """ Given an abstract transistor create its layout.
    
    :param t: 
    :param loc: 
    :param tech: module containing technology information
    :return: 
    """

    # Bottom left of l_active.
    x, y = loc

    # Choose l_active width such that at least one contact can be placed on each side of the transistor.
    w = tech.unit_cell_width + tech.via_size[l_diff_contact] + 2 * tech.minimum_enclosure[(l_active, l_diff_contact)]

    h = int(t.channel_width / tech.db_unit)

    x_eff = (x + 1) * tech.unit_cell_width - w // 2
    y_eff = 0
    if y % 2 == 1:
        # Top aligned.
        y_eff = y * tech.unit_cell_height - 2 * tech.routing_grid_pitch_y
        y_eff = grid_ceil(y_eff, tech.routing_grid_pitch_y, tech.grid_offset_y) + tech.via_size[l_diff_contact] // 2 + \
                tech.minimum_enclosure[(l_active, l_diff_contact)]
        y_eff = y_eff - h
    else:
        # Bottom aligned
        y_eff = y * tech.unit_cell_height + 1 * tech.routing_grid_pitch_y
        y_eff = grid_ceil(y_eff, tech.routing_grid_pitch_y, tech.grid_offset_y) - tech.via_size[l_diff_contact] // 2 - \
                tech.minimum_enclosure[(l_active, l_diff_contact)]

    active_box = pya.Box(
        x_eff,
        y_eff,
        x_eff + w,
        y_eff + h
    )

    nwell_box = None
    # Enclose active regions of PMOS transistors with l_nwell.
    if t.channel_type == ChannelType.PMOS:
        nwell2active_overlap = tech.minimum_enclosure[(l_nwell, l_active)]
        if not isinstance(nwell2active_overlap, tuple):
            nwell2active_overlap = (nwell2active_overlap, nwell2active_overlap)
        nwell2active_overlap_x, nwell2active_overlap_y = nwell2active_overlap

        nwell_box = pya.Box(
            x_eff - nwell2active_overlap_x,
            y_eff - nwell2active_overlap_y,
            x_eff + w + nwell2active_overlap_x,
            y_eff + h + nwell2active_overlap_y
        )

    center_x = active_box.center().x
    assert (center_x - tech.grid_offset_x) % tech.routing_grid_pitch_x == 0, Exception("Gate not x-aligned on grid.")

    source_box = pya.Box(
        x_eff,
        y_eff,
        center_x - tech.gate_length // 2,
        y_eff + h
    )

    drain_box = pya.Box(
        center_x - tech.gate_length // 2,
        y_eff,
        x_eff + w,
        y_eff + h
    )

    top = active_box.top
    bottom = active_box.bottom

    gate_top = top + tech.gate_extension
    gate_bottom = bottom - tech.gate_extension

    # Create gate terminals.
    terminals = {
        t.gate: [
            (l_poly, (center_x, gate_top)),
            (l_poly, (center_x, gate_bottom))
        ]
    }

    # Create gate shape.
    gate_path = pya.Path.new([
        pya.Point(center_x, gate_top),
        pya.Point(center_x, gate_bottom)],
        tech.gate_length,
        0,
        0)

    return TransistorLayout(gate_path, active_box, source_box, drain_box, terminals, nwell=nwell_box)
