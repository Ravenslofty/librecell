##
## Copyright (c) 2019 Thomas Kramer.
## 
## This file is part of librecell-lib 
## (see https://codeberg.org/tok/librecell/src/branch/master/librecell-lib).
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.
##
from liberty.parser import parse_liberty
from liberty.types import *

from ..logic.util import is_unate_in_xi
from ..liberty.util import get_pin_information

from .util import *
from .timing_combinatorial import characterize_comb_cell
from .input_capacitance import characterize_input_capacitances
import argparse

from copy import deepcopy
from PySpice.Unit import *
from PySpice.Spice.Parser import SpiceParser
import logging


def main():
    """
    Command-line tool for cell characterization.
    Currently only combinatorial cells are supported excluding tri-state cells.
    :return:
    """

    logger = logging.getLogger(__name__)
    logger.info("lctime main function")

    parser = argparse.ArgumentParser(
        description='Characterize the timing of a combinatorial cell based on a SPICE netlist. '
                    'The resulting liberty file will contain the data of the input liberty file '
                    'plus the updated characteristics of the selected cell.',
        epilog='Example: lctime --liberty specification.lib --cell INVX1 AND2X1 --spice netlists.sp -I transistor_model.m --output mylib.lib')

    parser.add_argument('-l', '--liberty', required=True, metavar='LIBERTY', type=str,
                        help='Liberty file. This must contain all necessary specifications needed to characterize the cell.')

    parser.add_argument('--cell', required=True, metavar='CELL_NAME', type=str,
                        action='append',
                        nargs='+',
                        help='Names of cells to be characterized.')

    parser.add_argument('--spice', required=True, metavar='SPICE', type=str,
                        action='append',
                        nargs='+',
                        help='SPICE netlist containing a subcircuit with the same name as the cell.')

    parser.add_argument('-I', '--include', required=False, action='append', metavar='SPICE_INCLUDE', type=str,
                        help='SPICE files to include such as transistor models.')

    parser.add_argument('--calc-mode', metavar='CALC_MODE', type=str, choices=['worst', 'typical', 'best'],
                        default='typical',
                        help='Calculation mode for computing the default timing arc based on the conditional timing arcs. "worst", "typical" (average) or "best".')

    parser.add_argument('-o', '--output', required=True, metavar='LIBERTY_OUT', type=str, help='Output liberty file.')

    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')

    # Parse arguments
    args = parser.parse_args()

    DEBUG = args.debug
    log_level = logging.DEBUG if DEBUG else logging.INFO

    if DEBUG:
        log_format = '%(module)16s %(levelname)8s: %(message)s'
    else:
        # Also output name of function in DEBUG mode.
        log_format = '%(module)16s %(funcName)16s %(levelname)8s: %(message)s'

    logging.basicConfig(format=log_format, level=log_level)

    # Get list of cell names to be characterized.
    cell_names = [n for names in args.cell for n in names]  # Flatten the nested list.

    # Get list of user-provided netlist files.
    netlist_files = [n for names in args.spice for n in names]  # Flatten the nested list.

    # Generate a lookup-table which tells for each cell name which netlist file to use.
    netlist_file_table: Dict[str, str] = dict()
    for netlist_file in netlist_files:
        logger.info("Load SPICE netlist: {}".format(netlist_file))
        parser = SpiceParser(path=netlist_file)
        for sub in parser.subcircuits:
            if sub.name in netlist_file_table:
                # Abort if a sub circuit is defined in multiple netlists.
                logger.warning(
                    f"Sub-circuit '{sub.name}' is defined in multiple netlists: {netlist_file_table[sub.name]}, {netlist_file}")
                exit(1)
            netlist_file_table[sub.name] = netlist_file

    # Test if all cell names can be found in the netlist files.
    cell_names_not_found = set(cell_names) - netlist_file_table.keys()
    if cell_names_not_found:
        logger.error("Cell name not found in netlists: {}".format(", ".join(cell_names_not_found)))
        exit(1)

    # Load liberty file.
    lib_file = args.liberty
    logger.info("Reading liberty: {}".format(lib_file))
    with open(lib_file) as f:
        data = f.read()
    library = parse_liberty(data)

    # Check if the delay model is supported.
    delay_model = library['delay_model']
    supported_delay_models = ['table_lookup']
    if delay_model not in supported_delay_models:
        msg = "Delay model not supported: '{}'. Must be one of {}.".format(delay_model,
                                                                           ", ".join(supported_delay_models))
        logger.error(msg)
        assert False, msg

    # Make independent copies of the library.
    new_library = deepcopy(library)
    # Strip all cell groups.
    new_library.groups = [g for g in new_library.groups if g.group_name != 'cell']

    # Load operation voltage and temperature.
    # TODO: load voltage/temperature from operating_conditions group
    supply_voltage = library['nom_voltage']
    temperature = library['nom_temperature']
    logger.info('Supply voltage = {:f} V'.format(supply_voltage))
    logger.info('Temperature = {:f} V'.format(temperature))

    # Units
    # TODO: choose correct unit from liberty file
    capacitance_unit_scale_factor = 1e12
    # TODO: get correct unit from liberty file.
    time_unit_scale_factor = 1e9

    # Get timing corner from liberty file.

    # Find definitions of operating conditions and sort them by name.
    operating_conditions_list = library.get_groups('operating_conditions')
    # Put into a dict by name.
    operating_conditions: Dict[str, Group] = {g.args[0]: g for g in operating_conditions_list}

    logger.info("Operating conditions: {}".format(set(operating_conditions.keys())))

    """
    TODO: Use the information from the operating conditions.
    Example:
    operating_conditions (MPSS) {
        calc_mode : worst ;
        process : 1.5 ;
        process_label : "ss" ;
        temperature : 70 ;
        voltage : 4.75 ;
        tree_type : worse_case_tree ;
    }
    """

    # TODO: let user overwrite it.
    calc_modes = {
        'typical': CalcMode.TYPICAL,
        'worst': CalcMode.WORST,
        'best': CalcMode.BEST,
    }

    # TODO: Make use of this.
    default_operating_conditions = library['default_operating_conditions']
    logger.info("Default operating conditions: {}".format(default_operating_conditions))

    assert args.calc_mode in calc_modes, "Unknown calculation mode: {}".format(args.calc_mode)

    calc_mode = calc_modes[args.calc_mode]
    logger.info("calc_mode: {}".format(calc_mode.name))

    # Read trip points from liberty file.
    trip_points = read_trip_points_from_liberty(library)

    logger.debug(trip_points)

    spice_includes = args.include if args.include else []
    if len(spice_includes) == 0:
        logger.warning("No transistor model supplied. Use --include or -I.")

    # Characterize all cells in the list.
    for cell_name in cell_names:
        netlist_file = netlist_file_table[cell_name]
        cell_group = select_cell(library, cell_name)
        assert cell_group.args[0] == cell_name
        logger.info("Cell: {}".format(cell_name))
        logger.info("Netlist: {}".format(netlist_file))

        # Get information on pins
        input_pins, output_pins, output_functions = get_pin_information(cell_group)

        if len(input_pins) == 0:
            msg = "Cell has no input pins."
            logger.error(msg)
            assert False, msg

        if len(output_pins) == 0:
            msg = "Cell has no output pins."
            logger.error(msg)
            assert False, msg

        # Add groups for the cell to be characterized.
        new_cell_group = deepcopy(select_cell(library, cell_name))
        # Strip away timing groups.
        for pin_group in new_cell_group.get_groups('pin'):
            pin_group.groups = [g for g in pin_group.groups if g.group_name != 'timing']
        new_library.groups.append(new_cell_group)

        logger.info("Run characterization")

        time_resolution = 50 @ u_ps
        logger.info("Time resolution = {}".format(time_resolution))

        # Measure input pin capacitances.
        logger.debug("Measuring input pin capacitances.")
        for input_pin in input_pins:
            logger.info("Measuring input capacitance: {}".format(input_pin))
            input_pin_group = new_cell_group.get_group('pin', input_pin)

            result = characterize_input_capacitances(
                cell_name=cell_name,
                input_pins=input_pins,
                active_pin=input_pin,
                output_pins=output_pins,
                supply_voltage=supply_voltage,
                trip_points=trip_points,
                timing_corner=calc_mode,
                spice_netlist_file=netlist_file_table[cell_name],
                spice_include_files=spice_includes,

                time_resolution=time_resolution,
                temperature=temperature
            )

            input_pin_group['rise_capacitance'] = result['rise_capacitance'] * capacitance_unit_scale_factor
            input_pin_group['fall_capacitance'] = result['fall_capacitance'] * capacitance_unit_scale_factor
            input_pin_group['capacitance'] = result['capacitance'] * capacitance_unit_scale_factor

        # Measure timing for all input-output arcs.
        logger.debug("Measuring timing.")
        for output_pin in output_pins:
            output_pin_group = new_cell_group.get_group('pin', output_pin)
            for related_pin in input_pins:
                logger.info("Timing arc: {} -> {}".format(related_pin, output_pin))

                # Get timing sense of this arc.
                timing_sense = is_unate_in_xi(output_functions[output_pin], related_pin).name.lower()
                logger.info("Timing sense: {}".format(timing_sense))

                result = characterize_comb_cell(
                    cell_name=cell_name,
                    input_pins=input_pins,
                    output_pin=output_pin,
                    related_pin=related_pin,
                    output_functions=output_functions,
                    supply_voltage=supply_voltage,
                    trip_points=trip_points,
                    timing_corner=calc_mode,
                    spice_netlist_file=netlist_file_table[cell_name],
                    spice_include_files=spice_includes,

                    time_resolution=time_resolution,
                    temperature=temperature
                )

                # TODO: get correct index/variable mapping from liberty file.
                index_1 = result['total_output_net_capacitance']
                index_2 = result['input_net_transition']
                # TODO: remember all necessary templates and create template tables.
                table_template_name = 'delay_template_{}x{}'.format(len(index_1), len(index_2))

                timing_tables = []
                for table_name in ['cell_rise', 'cell_fall', 'rise_transition', 'fall_transition']:
                    table = Group(
                        table_name,
                        args=[table_template_name],
                    )

                    table.set_array('index_1', index_1)
                    table.set_array('index_2', index_2)
                    table.set_array('values', result[table_name] * time_unit_scale_factor)

                    timing_tables.append(table)

                timing_group = Group(
                    'timing',
                    attributes={
                        'related_pin': EscapedString(related_pin),
                        'timing_sense': timing_sense
                    },
                    groups=timing_tables
                )

                # Attach timing group to output pin group.
                output_pin_group.groups.append(timing_group)

    with open(args.output, 'w') as f:
        logger.info("Write liberty: {}".format(args.output))
        f.write(str(new_library))
