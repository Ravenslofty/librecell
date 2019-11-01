import logging
import time
from typing import Dict, List, Tuple, Union
from klayout import db
import os

from .writer import Writer
from ..layout import layers
from ..lef import types as lef

logger = logging.getLogger(__name__)


def generate_lef_macro(cell_name: str,
                       pin_geometries: Dict[str, List[Tuple[str, db.Shape]]],
                       pin_direction: Dict[str, lef.Direction],
                       pin_use: Dict[str, lef.Use]
                       ) -> lef.Macro:
    """
    Assemble a LEF MACRO structure containing the pin shapes.
    :param cell_name: Name of the cell as it will appear in the LEF file.
    :param pin_geometries: A dictionary mapping pin names to geometries: Dict[pin name, List[(layer name, klayout Shape)]]
    :param pin_direction:
    :param pin_use:
    :return: Returns a `lef.Macro` object containing the pin information of the cell.

    # TODO: FOREIGN statement (reference to GDS)
    """

    logger.debug("Generate LEF MACRO structure for {}.".format(cell_name))
    pins = []
    # Create LEF Pin objects containing geometry information of the pins.
    for pin_name, ports in pin_geometries.items():

        layers = []

        for layer_name, shape in ports:
            # Convert all non-regions into a region
            region = db.Region()
            region.insert(shape)
            region.merge()

            geometries = []
            for p in region.each_merged():
                polygon = p.to_simple_polygon()

                box = polygon.bbox()
                is_box = db.SimplePolygon(box) == polygon

                if is_box:
                    rect = lef.Rect((box.p1.x, box.p1.y), (box.p2.x, box.p2.y))
                    geometries.append(rect)
                else:
                    # Port is a polygon
                    # Convert `db.Point`s into LEF points.
                    points = [(p.x, p.y) for p in polygon.each_point()]
                    poly = lef.Polygon(points)
                    geometries.append(poly)

            layers.append((lef.Layer(layer_name), geometries))

        port = lef.Port(CLASS=lef.Class.CORE,
                        geometries=layers)

        # if pin_name not in pin_direction:
        #     msg = "I/O direction of pin '{}' is not defined.".format(pin_name)
        #     logger.error(msg)
        #     assert False, msg
        #
        # if pin_name not in pin_use:
        #     msg = "Use of pin '{}' is not defined. Must be one of (CLK, SIGNAL, POWER, ...)".format(pin_name)
        #     logger.error(msg)
        #     assert False, msg

        pin = lef.Pin(pin_name=pin_name,
                      direction=lef.Direction.INOUT,  # TODO: find direction
                      use=lef.Use.SIGNAL,  # TODO: correct use
                      shape=lef.Shape.ABUTMENT,
                      port=port,
                      property={},
                      )
        pins.append(pin)

    macro = lef.Macro(
        name=cell_name,
        macro_class=lef.MacroClass.CORE,
        foreign=lef.Foreign(cell_name, lef.Point(0, 0)),
        origin=lef.Point(0, 0),
        symmetry={lef.Symmetry.X, lef.Symmetry.Y, lef.Symmetry.R90},
        site="CORE",
        pins=pins,
        obstructions=[]
    )

    return macro


class LefWriter(Writer):

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

        def remap_layers(layout: db.Layout) -> db.Layout:
            """
            Rename layer to match the scheme defined in the technology file.
            :param layout:
            :return:
            """
            logger.info("Remap layers.")
            layout2 = db.Layout()
            top1 = layout.top_cell()
            top2 = layout2.create_cell(top_cell.name)
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

        # Re-map layers
        layout = remap_layers(layout)

        # Set database unit.
        # klayout expects dbu to be in µm, the tech file takes it in meters.
        layout.dbu = self.db_unit * 1e6
        logger.info("dbu = {} µm".format(layout.dbu))

        # Possibly scale the layout.
        scaling_factor = 1
        if scaling_factor != 1:
            logger.info("Scaling layout by factor {}".format(scaling_factor))
            layout.transform(db.DCplxTrans(scaling_factor))

        # # Write GDS.
        # gds_file_name = '{}.gds'.format(top_cell.name)
        # gds_out_path = os.path.join(output_dir, gds_file_name)
        # logger.info("Write GDS: %s", gds_out_path)
        # layout.write(gds_out_path)

        # Write LEF
        # Create and populate LEF Macro data structure.
        # TODO: pass correct USE and DIRECTION
        lef_macro = generate_lef_macro(top_cell.name,
                                       pin_geometries=pin_geometries,
                                       pin_use=None,
                                       pin_direction=None)

        # Write LEF
        lef_file_name = "{}.lef".format(top_cell.name)
        lef_output_path = os.path.join(output_dir, lef_file_name)

        with open(lef_output_path, "w") as f:
            logger.info("Write LEF: {}".format(lef_output_path))
            f.write(lef.lef_format(lef_macro))
