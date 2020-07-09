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
import klayout.db as pya

import argparse
import logging

from .. import tech_util

from .drc_cleaner import clean

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='DRC cleaning.')
    parser.add_argument('-i', '--input', required=True, metavar='GDS', type=str, help='GDS input file')
    parser.add_argument('-o', '--output', required=True, metavar='GDS', type=str, help='GDS output file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--tech', required=True, metavar='FILE', type=str, help='technology file')
    parser.add_argument('--solver', default='z3', metavar='SOLVER', type=str, help='SMT solver name')
    parser.add_argument('--optimize', default=False, action='store_true', help='Enable area optimizations')

    # Parse arguments
    args = parser.parse_args()

    DEBUG = args.debug

    log_level = logging.DEBUG if DEBUG else logging.INFO

    logging.basicConfig(format='%(module)16s %(levelname)8s: %(message)s', level=log_level)

    layout = pya.Layout()
    logger.info("Reading GDS: %s", args.input)
    layout.read(args.input)
    top = layout.top_cell()

    tech = tech_util.load_tech_file(args.tech)

    # Fetch layers and create `shapes` object.
    # shapes: Dict[layer name, pya.Shapes]
    shapes = {}
    for name, (num, purpose) in tech.layermap.items():
        layer = layout.find_layer(num, purpose)
        # assert layer is not None, Exception('Layer not found: %s (%d, %d)' % (name, num, purpose))
        if layer is not None:
            shapes[name] = top.shapes(layer)

    logger.debug("shapes = %s", shapes)

    # Select all shapes that are completely inside `outline` on whitelisted layers
    # to be modified. (For instance excludes power rails from being modified.)
    white_list = set()
    outline = pya.Region(shapes['outline'])
    for layer in ['m1', 'm2', 'v1']:
        r = pya.Region(shapes[layer])
        inside = r.inside(outline)
        for shape in inside.each():
            white_list.add((layer, shape))

    drc_clean_success = clean(tech,
                              shapes,
                              white_list=white_list,
                              debug=DEBUG,
                              solver_name=args.solver,
                              optimize=args.optimize)

    if drc_clean_success:
        # Store result
        logger.info('Write GDS: %s', args.output)
        layout.write(args.output)
    else:
        logger.error("Could not clean design. GDS left unmodified.")
