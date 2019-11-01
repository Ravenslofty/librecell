import logging
import time
from typing import Dict, List, Tuple
from klayout import db
import os

from .writer import Writer, remap_layers
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
