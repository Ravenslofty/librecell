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
import logging
import time
from typing import Dict, List, Tuple, Union
from klayout import db
import os

from .writer import Writer, remap_layers
from ..layout import layers

logger = logging.getLogger(__name__)


class GdsWriter(Writer):
    """
    Writer for GDS2 output.
    """

    def __init__(self,
                 db_unit: float,
                 output_map: Dict[str, Tuple[int, int]]):
        self.db_unit = db_unit
        self.output_map = output_map

    def write_layout(self,
                     layout: db.Layout,
                     pin_geometries: Dict[str, List[Tuple[str, db.Shape]]],
                     top_cell: db.Cell,
                     output_dir: str,
                     ) -> None:
        # Re-map layers
        layout = remap_layers(layout, self.output_map)

        # Set database unit.
        # klayout expects dbu to be in µm, the tech file takes it in meters.
        layout.dbu = self.db_unit * 1e6
        logger.debug("dbu = {} µm".format(layout.dbu))

        # Possibly scale the layout.
        scaling_factor = 1
        if scaling_factor != 1:
            logger.info("Scaling layout by factor {}".format(scaling_factor))
            layout.transform(db.DCplxTrans(scaling_factor))

        # Write GDS.
        gds_file_name = '{}.gds'.format(top_cell.name)
        gds_out_path = os.path.join(output_dir, gds_file_name)
        logger.info("Write GDS: %s", gds_out_path)
        layout.write(gds_out_path)
