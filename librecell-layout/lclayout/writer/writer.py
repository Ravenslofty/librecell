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
from ..layout import layers
from klayout import db
from typing import Dict, List, Tuple, Union

import logging

logger = logging.getLogger(__name__)


class Writer:

    def write_layout(self,
                     layout: db.Layout,
                     pin_geometries: Dict[str, List[Tuple[str, db.Shape]]],
                     top_cell: db.Cell,
                     output_dir: str,
                     ) -> None:
        pass


def remap_layers(layout: db.Layout, output_map: Dict[str, Union[Tuple[int, int], List[Tuple[int, int]]]]) -> db.Layout:
    """
    Rename layer to match the scheme defined in the technology file.
    :param layout:
    :param output_map: Output mapping from layer names to layer numbers.
    :return:
    """
    logger.debug("Remap layers.")
    layout2 = db.Layout()

    for top1 in layout.each_cell():
        top2 = layout2.create_cell(top1.name)
        layer_infos1 = layout.layer_infos()
        for layer_info in layer_infos1:

            src_layer = (layer_info.layer, layer_info.datatype)

            if src_layer not in layers.layermap_reverse:
                msg = "Layer {} not defined in `layermap_reverse`.".format(src_layer)
                logger.warning(msg)
                dest_layers = src_layer
            else:
                src_layer_name = layers.layermap_reverse[src_layer]

                if src_layer_name not in output_map:
                    msg = "Layer '{}' will not be written to the output. This might be alright though.". \
                        format(src_layer_name)
                    logger.warning(msg)
                    continue

                dest_layers = output_map[src_layer_name]

            if not isinstance(dest_layers, list):
                dest_layers = [dest_layers]

            src_idx = layout.layer(layer_info)
            for dest_layer in dest_layers:
                dest_idx = layout2.layer(*dest_layer)
                top2.shapes(dest_idx).insert(top1.shapes(src_idx))

    return layout2
