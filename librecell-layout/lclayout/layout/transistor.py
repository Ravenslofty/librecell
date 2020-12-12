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
from ..place.place import Transistor, ChannelType

from .layers import *
from typing import Any, Dict, List, Optional, Set, Tuple
import sys

# klayout.db should not be imported if script is run from KLayout GUI.
if 'pya' not in sys.modules:
    import klayout.db as pya


class TransistorLayout:
    """ Implementations of this class are responsible for drawing a transistor to the layout.
    The function `draw()` must be implemented.
    """

    def __init__(self, abstract_transistor: Transistor, location: Tuple[int, int], distance_to_outline: int, tech):
        """
        Create the layout representation of a transistor based on the abstract transistor (netlist) a location within the cell
        and design rules.
        :param abstract_transistor: Netlist representation of the transistor.
        :param location: Location in the cell matrix.
        :param distance_to_outline: TODO: This should be put into the `tech`.
        :param tech: Technology specific designrules.
        """
        raise NotImplemented()

    def terminal_nodes(self) -> Dict[str, List[Tuple[str, Tuple[int, int]]]]:
        """
        Get point-like terminal nodes in the form `{net name: {(layer name, (x, y)), ...}}`.

        This function allows to define point-like terminals at precise locations additionally to the terminals
        defined by polygons in the layout.

        This could be used for instance if a net region does not touch any grid points. Hence it is possible to insert
        off-grid routing terminals.
        """
        return dict()

    def draw(self, shapes: Dict[Any, pya.Shapes]) -> None:
        """ Draw the TransistorLayout.

        Routing terminals must be labelled with the `'net'` property.

        Example
        =======
        To insert the gate of a transistor:

        `shapes[l_poly].insert(gate_shape).set_property('net', gate_net)`

        :param shapes: Dict[layer name, pya.Shapes]
          A dict mapping layer names to pya.Shapes.
        """
        raise NotImplemented()


class DefaultTransistorLayout(TransistorLayout):
    """ Layout representation of transistor.
    
    Contains the shapes of the transistor plus shapes that mark possible locations of contacts.
    """

    def __init__(self, abstract_transistor: Transistor, location: Tuple[int, int], tech):
        """
        Create the layout representation of a transistor based on the abstract transistor (netlist) a location within the cell
        and design rules.
        :param abstract_transistor: Netlist representation of the transistor.
        :param location: Location in the cell matrix.
        :param distance_to_outline: TODO: This should be put into the `tech`.
        :param tech: Technology specific designrules.
        """

        self.abstract_transistor = abstract_transistor
        self.location = location
        self.distance_to_outline = tech.transistor_offset_y # TODO: Simplify this.

        # Get either the ndiffusion or pdiffusion layer.
        if abstract_transistor.channel_type == ChannelType.NMOS:
            l_diffusion = l_ndiffusion
            l_diff_contact = l_ndiff_contact
        else:
            l_diffusion = l_pdiffusion
            l_diff_contact = l_pdiff_contact

        # Diffusion layer.
        self.l_diffusion = l_diffusion

        # Calculate minimal distance from active region to upper and lower cell boundaries.
        poly_half_spacing = (tech.min_spacing[(l_poly, l_poly)] + 1) // 2
        active_half_spacing = (tech.min_spacing[(l_diffusion, l_diffusion)] + 1) // 2
        # Distance from active to active in neighbouring cell must be kept,
        # as well as distance from poly to poly in neighbouring cell.
        min_distance_to_outline = max(active_half_spacing, tech.gate_extension + poly_half_spacing)

        assert self.distance_to_outline >= min_distance_to_outline, 'Chosen distance will violate minimum spacing rules. {} >= {}.'.format(
            self.distance_to_outline, min_distance_to_outline)

        # Bottom left of l_diffusion.
        x, y = location

        # Choose l_diffusion width such that at least one contact can be placed on each side of the transistor.
        w = tech.unit_cell_width + tech.via_size[l_diff_contact] + 2 * tech.minimum_enclosure[
            (l_diffusion, l_diff_contact)]

        h = abstract_transistor.channel_width

        x_eff = (x + 1) * tech.unit_cell_width - w // 2
        y_eff = 0
        if y % 2 == 1:
            # Top aligned.
            y_eff = y * tech.unit_cell_height - self.distance_to_outline
            # y_eff = grid_floor(y_eff, tech.routing_grid_pitch_y, tech.grid_offset_y) + tech.via_size[l_diff_contact] // 2 + \
            #         tech.minimum_enclosure[(l_diffusion, l_diff_contact)]
            y_eff = y_eff - h
        else:
            # Bottom aligned
            y_eff = y * tech.unit_cell_height + self.distance_to_outline
            # y_eff = grid_ceil(y_eff, tech.routing_grid_pitch_y, tech.grid_offset_y) - tech.via_size[l_diff_contact] // 2 - \
            #         tech.minimum_enclosure[(l_diffusion, l_diff_contact)]

        # Create shape for active layer.
        active_box = pya.Box(
            x_eff,
            y_eff,
            x_eff + w,
            y_eff + h
        )

        # Enclose active regions of PMOS transistors with l_nwell.

        # Get layer depending on channel type.
        l_well = l_nwell if abstract_transistor.channel_type == ChannelType.PMOS else l_pwell
        # Well layer.
        self.l_well = l_well

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

        center_x = active_box.center().x
        assert (center_x - tech.grid_offset_x) % tech.routing_grid_pitch_x == 0, Exception(
            "Gate not x-aligned on grid.")

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
            abstract_transistor.gate_net: [
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

        self._gate_path = gate_path
        self._active_box = active_box
        self._source_box = source_box
        self._drain_box = drain_box
        self._terminals = terminals
        self._well_box = well_box

    def terminal_nodes(self) -> Dict[str, List[Tuple[str, Tuple[int, int]]]]:
        """
        Get point-like terminal nodes in the form `{net name: {(layer name, (x, y)), ...}}`.
        """
        return self._terminals

    def draw(self, shapes: Dict[Any, pya.Shapes]) -> None:
        """ Draw a TransistorLayout.

        :param shapes: Dict[layer name, pya.Shapes]
          A dict mapping layer names to pya.Shapes.
        """

        # Create well and active shape.
        shapes[self.l_well].insert(self._well_box)
        shapes[self.l_diffusion].insert(self._active_box)
        shapes[self.l_diffusion].insert(self._source_box).set_property('net', self.abstract_transistor.source_net)
        shapes[self.l_diffusion].insert(self._drain_box).set_property('net', self.abstract_transistor.drain_net)

        # Create gate shape.
        inst = shapes[l_poly].insert(self._gate_path)
        inst.set_property('net', self.abstract_transistor.gate_net)
