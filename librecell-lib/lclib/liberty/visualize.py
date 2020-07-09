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
from liberty.parser import parse_liberty

from liberty.types import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import argparse
import logging

logger = logging.getLogger(__name__)


def main_plot_timing():
    """
    Command-line tool to visualize NDLM timing models in liberty files.
    :return:
    """
    parser = argparse.ArgumentParser(description='Visualize NDLM timing arrays in liberty files.')
    parser.add_argument('-l', '--liberty', required=True, metavar='LIBERTY', type=str, help='Liberty file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--cell', required=True, metavar='CELL_NAME', type=str, help='Cell name.')
    parser.add_argument('--pin', required=True, metavar='PIN_NAME', type=str, help='Pin name.')
    parser.add_argument('--related-pin', required=True, metavar='RELATED_PIN_NAME', type=str,
                        help='Related pin name.')
    parser.add_argument('--timing-type', required=False, default=None, metavar='TIMING_TYPE', type=str,
                        help='Value of timing_type attribute.')

    parser.add_argument('--table', required=True, metavar='TABLE_NAME', type=str,
                        help='Type of table: cell_rise, cell_fall, rise_transition, fall_transition, ...')

    # Parse arguments
    args = parser.parse_args()

    DEBUG = args.debug
    log_level = logging.DEBUG if DEBUG else logging.INFO

    logging.basicConfig(format='%(module)16s %(levelname)8s: %(message)s', level=log_level)

    lib_file = args.liberty

    logger.info("Reading liberty: {}".format(lib_file))
    with open(lib_file) as f:
        data = f.read()

    library = parse_liberty(data)

    num_cells = len([g for g in library.get_groups('cell')])
    logger.info("Number of cells: {}".format(num_cells))

    cell = select_cell(library, args.cell)
    pin = select_pin(cell, args.pin)

    timing_table = select_timing_table(pin, related_pin=args.related_pin,
                                       timing_type=args.timing_type,
                                       table_name=args.table)

    plot_timing_ndlm(library, timing_table)


def plot3d(x_axis: np.ndarray, y_axis: np.ndarray, z_data: np.ndarray,
           x_label="",
           y_label="",
           z_label=""):
    """
    Show a 3D surface plot.
    :param x_axis:
    :param y_axis:
    :param z_data:
    :param x_label:
    :param y_label:
    :param z_label:
    :return:
    """
    xx, yy = np.meshgrid(x_axis, y_axis)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    ax.plot_surface(xx, yy, z_data)
    plt.show()


def plot_timing_ndlm(library: Group, table: Group):
    """
    Plot a NDLM timing table.
    :param library:
    :param table:
    :return:
    """

    x_axis = table.get_array('index_2')
    y_axis = table.get_array('index_1')
    z_data = table.get_array('values')

    template_name = table.args[0]
    template = library.get_group('lu_table_template', template_name)

    time_unit = library['time_unit'].value

    x_label = template['variable_2']
    y_label = template['variable_1']

    z_label = "[{}]".format(time_unit)

    plot3d(x_axis, y_axis, z_data, x_label=x_label, y_label=y_label, z_label=z_label)


def test_plot_nldm():
    import os.path
    lib_file = os.path.join(os.path.dirname(__file__), '../../test_data/gscl45nm.lib')

    data = open(lib_file).read()

    library = parse_liberty(data)

    cell = select_cell(library, 'INVX2')
    pin = select_pin(cell, 'Y')

    timing_table = select_timing_table(pin, 'A', 'rise_transition')

    plot_timing_ndlm(library, timing_table)
