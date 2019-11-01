import logging
import time
from typing import Dict, List, Tuple, Union
from klayout import db
import os

from .writer import Writer
from ..layout import layers

logger = logging.getLogger(__name__)


class GdsWriter(Writer):

    def __init__(self,
                 db_unit: float,
                 output_map: Dict[str, Tuple[int, int]]):
        self.db_unit = db_unit
        self.output_map = output_map

    def _remap_layers(self, layout: db.Layout) -> db.Layout:
        """
        Rename layer to match the scheme defined in the technology file.
        :param layout:
        :return:
        """
        logger.info("Remap layers.")
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

                    if src_layer_name not in self.output_map:
                        msg = "Layer '{}' will not be written to the output. This might be alright though.". \
                            format(src_layer_name)
                        logger.warning(msg)
                        continue

                    dest_layers = self.output_map[src_layer_name]

                if not isinstance(dest_layers, list):
                    dest_layers = [dest_layers]

                src_idx = layout.layer(layer_info)
                for dest_layer in dest_layers:
                    dest_idx = layout2.layer(*dest_layer)
                    top2.shapes(dest_idx).insert(top1.shapes(src_idx))

        return layout2

    def write_layout(self,
                     layout: db.Layout,
                     pin_geometries: Dict[str, List[Tuple[str, db.Shape]]],
                     top_cell: db.Cell,
                     output_dir: str,
                     ) -> None:

        # Re-map layers
        layout = self._remap_layers(layout)

        # Set database unit.
        # klayout expects dbu to be in µm, the tech file takes it in meters.
        layout.dbu = self.db_unit * 1e6
        logger.info("dbu = {} µm".format(layout.dbu))

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
