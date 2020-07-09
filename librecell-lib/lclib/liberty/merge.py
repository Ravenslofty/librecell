#
# Copyright (c) 2019-2020 Thomas Kramer.
#
# This file is part of librecell 
# (see https://codeberg.org/tok/librecell).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""
Tool for merging liberty libraries.
"""

from liberty.parser import parse_liberty
from liberty.types import Group
from typing import List
import argparse
import logging
from itertools import chain
from copy import copy
import os

logger = logging.getLogger(__name__)


def add_named_group(base_library: Group, replacement: Group, replace_if_already_exists: bool = False) -> None:
    """
    Add or replace a named group in the base library. The name of the group is `group.args[0]`.
    If the group does not yet exist it will be appended to the base library. If it exists, then
    depending on `replace_if_already_exists` the existing group will be replaced with the `replacement`.
    :param base_library: Group container to be updated.
    :param replacement:
    :param replace_if_already_exists:
    :return: None
    """
    cell_name = replacement.args[0]

    for i in range(len(base_library.groups)):
        g = base_library.groups[i]
        if g.group_name == replacement.group_name and g.args[0] == cell_name:
            if replace_if_already_exists:
                logger.info("Replace group: {}({})".format(replacement.group_name, replacement.args[0]))
                base_library.groups[i] = replacement
            return

    # If cell has not been replaced append the new cell.
    logger.info("Add group: {}({})".format(replacement.group_name, replacement.args[0]))
    base_library.groups.append(replacement)


def add_named_groups(base_library: Group, replacements: List[Group], replace_if_already_exists: bool = False) -> None:
    for repl in replacements:
        add_named_group(base_library, repl, replace_if_already_exists)


def main():
    """
    Command-line for merging liberty libraries.
    :return:
    """

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Merge liberty libraries. The content of the base library will'
                                                 'be replaced by the content in the update libraries.')

    parser.add_argument('-b', '--base', required=True, metavar='LIBERTY_BASE', type=str, help='Base liberty file.')

    parser.add_argument('-o', '--output', required=True, metavar='LIBERTY_OUT', type=str, help='Output liberty file.')

    parser.add_argument('-u', '--update', required=False, action='append', nargs='+', metavar='LIBERTY', type=str,
                        help='Liberty files with updates.')

    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--force', action='store_true', help='Allow overwriting of output file.')

    merge_modes = {'replace_cells'}

    parser.add_argument('--merge', default='replace_cells',
                        required=False, choices=merge_modes,
                        metavar='MERGE_MODE', type=str, help='Choose how to merge.')

    # Parse arguments
    args = parser.parse_args()

    DEBUG = args.debug
    log_level = logging.DEBUG if DEBUG else logging.INFO

    logging.basicConfig(format='%(levelname)8s: %(message)s', level=log_level)

    # Check if output would overwrite something.
    if not args.force and os.path.exists(args.output):
        logger.error("Output file exists. Use --force to overwrite it.")
        exit(1)

    # Read base liberty.
    base_lib_file = args.base
    with open(base_lib_file) as f:
        logger.info("Reading base liberty: {}".format(base_lib_file))
        data = f.read()
    base_library = parse_liberty(data)

    # Read updates
    update_libraries = []
    for lib_file in chain(*args.update):
        with open(lib_file) as f:
            logger.info("Reading liberty: {}".format(lib_file))
            data = f.read()
        lib = parse_liberty(data)
        update_libraries.append(lib)

    if args.merge == 'replace_cells':

        new_library = copy(base_library)

        # Add updates to new library.
        for lib in update_libraries:
            # Add cell groups
            add_named_groups(new_library, lib.get_groups('cell'), replace_if_already_exists=True)

            # Add non-existent lookup table groups
            add_named_groups(new_library, lib.get_groups('lu_table_template'), replace_if_already_exists=False)
            add_named_groups(new_library, lib.get_groups('power_lut_template'), replace_if_already_exists=False)

        num_cells_old = len(base_library.get_groups('cell'))
        num_cells_new = len(new_library.get_groups('cell'))

        logger.info("Number of cells in base: {}, number of cells in output: {}".format(num_cells_old, num_cells_new))

        with open(args.output, 'w') as f:
            logger.info('Write liberty: {}'.format(args.output))
            f.write(str(new_library))
