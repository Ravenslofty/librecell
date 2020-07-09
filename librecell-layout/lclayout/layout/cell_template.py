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
from .layers import *
from klayout import db
from typing import Dict, Tuple

"""
      # Fill half of the cell with nwell.
        # TODO: do this in the cell template or after placing the transistors.
        nwell_box = pya.Box(
            pya.Point(0, cell_height // 2),
            pya.Point(cell_width, cell_height)
        )

        shapes[l_nwell].insert(nwell_box)
"""


def draw_cell_template(shapes: Dict[str, db.Shapes],
                       cell_shape: Tuple[int, int],
                       nwell_pwell_spacing: int = 0) -> None:
    """
    Draw shapes of the cell that can be drawn without knowledge of the transistor placement.
    This includes the cell boundary, nwell/pwell.
    :param shapes: KLayout shapes that will be modified.
    :param cell_shape: Dimensions of the cell (width, height).
    :param nwell_pwell_spacing: Minimum spacing between nwell and pwell for twin-well layouts.
    :return: None
    """
    cell_width, cell_height = cell_shape

    # Draw abutment box.
    shapes[l_abutment_box].insert(db.Box(0, 0, cell_width, cell_height))

    # Draw nwell / pwell

    # Fill half of the cell with nwell.

    nwell_start_y = cell_height // 2 + nwell_pwell_spacing // 2
    nwell_end_y = cell_height

    pwell_start_y = 0
    pwell_end_y = cell_height // 2 - nwell_pwell_spacing // 2

    nwell_box = db.Box(
        db.Point(0, nwell_start_y),
        db.Point(cell_width, nwell_end_y)
    )

    shapes[l_nwell].insert(nwell_box)

    pwell_box = db.Box(
        db.Point(0, pwell_start_y),
        db.Point(cell_width, pwell_end_y)
    )
    shapes[l_pwell].insert(pwell_box)
