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
from .. import tech_util
from typing import Any, Dict, Tuple, Optional
import sys

# klayout.db should not be imported if script is run from KLayout GUI.
if 'pya' not in sys.modules:
    import klayout.db as pya


class TransistorLayout:
    """ Layout representation of transistor.
    
    Contains the shapes of the transistor plus shapes that mark possible locations of contacts.
    """

    def __init__(self, gate: pya.Box,
                 active: pya.Box,
                 source_box: pya.Box,
                 drain_box: pya.Box,
                 terminals,
                 nwell: Optional[pya.Box] = None,
                 pwell: Optional[pya.Box] = None):
        """

        :param gate: pya.Path
          Layout of PC gate.

        :param active: pya.Box
          Layout of the diffusion region.

        :param source_box: pya.Box
          Marks possible locations for contacts to source.

        :param drain_box: pya.Box
          Marks possible locations for contacts to drain.

        """

        assert (nwell is None) ^ (pwell is None), "Exactly one of {nwell, pwell} must be defined."

        # Transistor polygons/paths
        self.gate = gate
        self.active = active

        # n-well for P-mos
        self.nwell = nwell

        # p-well for N-mos
        self.pwell = pwell

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

    if t.nwell:
        # For PMOS.
        shapes[l_pdiffusion].insert(t.active)
        shapes[l_nwell].insert(t.nwell)

    if t.pwell:
        # For NMOS.
        shapes[l_ndiffusion].insert(t.active)
        shapes[l_pwell].insert(t.pwell)

    shapes[l_poly].insert(t.gate)


def create_transistor_layout(t: Transistor, loc: Tuple[int, int], distance_to_outline: int, tech) -> TransistorLayout:
    """ Given an abstract transistor create its layout.
    
    :param t: 
    :param loc: Transistor location on the grid. (0,0) is the transistor on the bottom left, (0,1) its upper neighbour, (1,0) its right neighbour.
    :param distance_to_outline: Distance of active area to upper or lower boundary of the cell. Basically determines the y-offset of the transistors.
    :param tech: module containing technology information
    :return: 
    """

    # Get either the ndiffusion or pdiffusion layer.
    if t.channel_type == ChannelType.NMOS:
        l_diffusion = l_ndiffusion
    else:
        l_diffusion = l_pdiffusion

    # Calculate minimal distance from active region to upper and lower cell boundaries.
    poly_half_spacing = (tech.min_spacing[(l_poly, l_poly)] + 1) // 2
    active_half_spacing = (tech.min_spacing[(l_diffusion, l_diffusion)] + 1) // 2
    # Distance from active to active in neighbouring cell must be kept,
    # as well as distance from poly to poly in neighbouring cell.
    min_distance_to_outline = max(active_half_spacing, tech.gate_extension + poly_half_spacing)

    assert distance_to_outline >= min_distance_to_outline, 'Chosen distance will violate minimum spacing rules. {} >= {}.'.format(
        distance_to_outline, min_distance_to_outline)

    # Bottom left of l_diffusion.
    x, y = loc

    # Choose l_diffusion width such that at least one contact can be placed on each side of the transistor.
    w = tech.unit_cell_width + tech.via_size[l_diff_contact] + 2 * tech.minimum_enclosure[(l_diffusion, l_diff_contact)]

    h = t.channel_width

    x_eff = (x + 1) * tech.unit_cell_width - w // 2
    y_eff = 0
    if y % 2 == 1:
        # Top aligned.
        y_eff = y * tech.unit_cell_height - distance_to_outline
        # y_eff = grid_floor(y_eff, tech.routing_grid_pitch_y, tech.grid_offset_y) + tech.via_size[l_diff_contact] // 2 + \
        #         tech.minimum_enclosure[(l_diffusion, l_diff_contact)]
        y_eff = y_eff - h
    else:
        # Bottom aligned
        y_eff = y * tech.unit_cell_height + distance_to_outline
        # y_eff = grid_ceil(y_eff, tech.routing_grid_pitch_y, tech.grid_offset_y) - tech.via_size[l_diff_contact] // 2 - \
        #         tech.minimum_enclosure[(l_diffusion, l_diff_contact)]

    # Create shape for active layer.
    active_box = pya.Box(
        x_eff,
        y_eff,
        x_eff + w,
        y_eff + h
    )

    nwell_box = None
    pwell_box = None

    # Enclose active regions of PMOS transistors with l_nwell.

    # Get layer depending on channel type.
    l_well = l_nwell if t.channel_type == ChannelType.PMOS else l_pwell

    # Get minimum overlap from tech file.
    well2active_overlap = tech.minimum_enclosure.get((l_well, l_diffusion), 0)
    if not isinstance(well2active_overlap, tuple):
        well2active_overlap = (well2active_overlap, well2active_overlap)
    well2active_overlap_x, well2active_overlap_y = well2active_overlap

    # Create shape of nwell or pwell.
    well_box = pya.Box(
        x_eff - well2active_overlap_x,
        y_eff - well2active_overlap_y,
        x_eff + w + well2active_overlap_x,
        y_eff + h + well2active_overlap_y
    )

    if t.channel_type == ChannelType.PMOS:
        nwell_box = well_box
    elif t.channel_type == ChannelType.NMOS:
        pwell_box = well_box

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

    return TransistorLayout(gate_path, active_box, source_box, drain_box, terminals,
                            nwell=nwell_box,
                            pwell=pwell_box)
