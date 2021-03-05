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
from typing import Dict, List, Tuple, Optional
from klayout import db
import os

from .writer import Writer, remap_layers
from ..layout import layers
from ..lef import types as lef

logger = logging.getLogger(__name__)


def _decompose_region(region: db.Region, ignore_non_rectilinear: bool = False) -> List[db.Box]:
    """
    Decompose a `db.Region` of multiple `db.Polygon`s into non-overlapping rectangles (`db.Box`).
    :param region:
    :param ignore_non_rectilinear: If set to `True` then non-rectilinear polygons are skipped.
    :return: Returns the list of rectangles.
    """
    trapezoids = region.decompose_trapezoids_to_region()
    logger.debug("Number of trapezoids: {}".format(trapezoids.size()))
    rectangles = []
    for polygon in trapezoids.each():
        box = polygon.bbox()

        if db.Polygon(box) != polygon:
            msg = "Cannot decompose into rectangles. Something is not rectilinear!"
            if not ignore_non_rectilinear:
                logger.error(msg)
                assert False, msg
            else:
                logger.warning(msg)

        rectangles.append(box)
    return rectangles


def generate_lef_macro(cell_name: str,
                       pin_geometries: Dict[str, List[Tuple[str, db.Shape]]],
                       pin_direction: Dict[str, lef.Direction],
                       pin_use: Dict[str, lef.Use],
                       site: str = "CORE",
                       scaling_factor: float = 1,
                       use_rectangles_only: bool = False,
                       ) -> lef.Macro:
    """
    Assemble a LEF MACRO structure containing the pin shapes.
    :param site: SITE name. Default is 'CORE'.
    :param cell_name: Name of the cell as it will appear in the LEF file.
    :param pin_geometries: A dictionary mapping pin names to geometries: Dict[pin name, List[(layer name, klayout Shape)]]
    :param pin_direction:
    :param pin_use:
    :param use_rectangles_only: Decompose all polygons into rectangles. Non-rectilinear shapes are dropped.
    :return: Returns a `lef.Macro` object containing the pin information of the cell.

    # TODO: FOREIGN statement (reference to GDS)
    """

    logger.debug("Generate LEF MACRO structure for {}.".format(cell_name))
    logger.debug(f"Scaling factor = {scaling_factor}.")

    f = scaling_factor

    pins = []
    # Create LEF Pin objects containing geometry information of the pins.
    for pin_name, ports in pin_geometries.items():

        layers = []

        for layer_name, shape in ports:
            # Convert all non-regions into a region
            region = db.Region()
            region.insert(shape)
            region.merge()
            if use_rectangles_only:
                # Decompose into rectangles.
                boxes = _decompose_region(region)
                region = db.Region()
                region.insert(boxes)

            geometries = []
            for p in region.each():
                polygon = p.to_simple_polygon()

                box = polygon.bbox()
                is_box = db.SimplePolygon(box) == polygon

                if is_box:
                    rect = lef.Rect((box.p1.x * f, box.p1.y * f), (box.p2.x * f, box.p2.y * f))
                    geometries.append(rect)
                else:
                    # Port is a polygon
                    # Convert `db.Point`s into LEF points.
                    points = [(p.x * f, p.y * f) for p in polygon.each_point()]
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
        site=site,
        pins=pins,
        obstructions=[]
    )

    return macro


class LefWriter(Writer):

    def __init__(self,
                 output_map: Dict[str, Tuple[int, int]],
                 site: str = "CORE",
                 db_unit: float = 1e-6,
                 use_rectangles_only: bool = False):
        """

        :param output_map:
        :param site: SITE name.
        :param db_unit: Database unit in meters. Default is 1um (1e-6 m)
        :param use_rectangles_only: Convert all polygons into rectangles. Non-rectilinear shapes are dropped.
        """
        self.db_unit = db_unit
        self.output_map = output_map
        self.scaling_factor = 1
        self.use_rectangles_only = use_rectangles_only
        self.site = site

    def write_layout(self,
                     layout: db.Layout,
                     pin_geometries: Dict[str, List[Tuple[str, db.Shape]]],
                     top_cell: db.Cell,
                     output_dir: str,
                     ) -> None:
        # Re-map layers
        layout = remap_layers(layout, self.output_map)

        # Compute correct scaling factor.
        # klayout expects dbu to be in µm, the tech file takes it in meters.
        logger.debug(f"LEF db_unit = {self.db_unit} [m]")
        scaling_factor = self.db_unit / (layout.dbu)
        scaling_factor *= self.scaling_factor  # Allow to make corrections from the tech file.

        # Write LEF
        # Create and populate LEF Macro data structure.
        # TODO: pass correct USE and DIRECTION
        lef_macro = generate_lef_macro(top_cell.name,
                                       pin_geometries=pin_geometries,
                                       pin_use=None,
                                       pin_direction=None,
                                       site=self.site,
                                       scaling_factor=scaling_factor,
                                       use_rectangles_only=self.use_rectangles_only)

        # Write LEF
        lef_file_name = "{}.lef".format(top_cell.name)
        lef_output_path = os.path.join(output_dir, lef_file_name)

        with open(lef_output_path, "w") as f:
            logger.info("Write LEF: {}".format(lef_output_path))
            f.write(lef.lef_format(lef_macro))
