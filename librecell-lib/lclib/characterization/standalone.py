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

from liberty import parser as liberty_parser
from liberty import boolean_functions as liberty_bools
from liberty.types import *
from liberty.arrays import *

from ..logic.util import is_unate_in_xi
from ..liberty import util as liberty_util
from ..logic import functional_abstraction

from .util import *
from .timing_combinatorial import characterize_comb_cell
from .input_capacitance import characterize_input_capacitances
import argparse
import logging
from copy import deepcopy
from PySpice.Unit import *
from lccommon import net_util
from lccommon.net_util import load_transistor_netlist, is_ground_net, is_supply_net
import networkx as nx
import sympy
from typing import Iterable


def _boolean_to_lambda(boolean: sympy.boolalg.Boolean):
    """
    Convert a sympy.boolalg.Boolean expression into a Python lambda function.
    :param boolean:
    :return:
    """
    simple = sympy.simplify(boolean)
    f = sympy.lambdify(boolean.atoms(), simple)
    return f


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
        epilog='Example: lctime --liberty specification.lib --cell INVX1 --spice INVX1.sp -I transistor_model.m --output mylib.lib')

    parser.add_argument('-l', '--liberty', required=True, metavar='LIBERTY', type=str,
                        help='Liberty file. This must contain all necessary specifications needed to characterize the cell.')

    parser.add_argument('--cell', required=True, metavar='CELL_NAME', type=str, help='Cell name.')

    parser.add_argument('--spice', required=True, metavar='SPICE', type=str,
                        help='SPICE netlist containing a subcircuit with the same name as the cell.')

    parser.add_argument('-I', '--include', required=False, action='append', metavar='SPICE_INCLUDE', type=str,
                        help='SPICE files to include such as transistor models.')

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

    cell_name = args.cell
    lib_file = args.liberty

    logger.info("Reading liberty: {}".format(lib_file))
    with open(lib_file) as f:
        data = f.read()

    library = liberty_parser.parse_liberty(data)

    # Check if the delay model is supported.
    delay_model = library['delay_model']
    supported_delay_models = ['table_lookup']
    if delay_model not in supported_delay_models:
        msg = "Delay model not supported: '{}'. Must be one of {}.".format(delay_model,
                                                                           ", ".join(supported_delay_models))
        logger.error(msg)
        assert False, msg

    new_library = deepcopy(library)
    # Strip all cell groups.
    new_library.groups = [g for g in new_library.groups if g.group_name != 'cell']
    # Add group of current cell.
    new_cell_group = deepcopy(select_cell(library, cell_name))
    # Strip away timing groups.
    for pin_group in new_cell_group.get_groups('pin'):
        pin_group.groups = [g for g in pin_group.groups if g.group_name != 'timing']
    new_library.groups.append(new_cell_group)

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

    cell_group = select_cell(library, cell_name)
    assert cell_group.args[0] == cell_name
    logger.info("Cell: {}".format(cell_name))

    # Load netlist of cell
    netlist_path = args.spice
    logger.info('Load netlist: %s', netlist_path)
    transistors_abstract, cell_pins = load_transistor_netlist(netlist_path, cell_name)
    io_pins = net_util.get_io_pins(cell_pins)

    # Detect power pins.
    # TODO: don't decide based only on net name.
    power_pins = [p for p in cell_pins if net_util.is_power_net(p)]
    assert len(power_pins) == 2, "Expected to have 2 power pins."
    vdd_pins = [p for p in power_pins if net_util.is_supply_net(p)]
    gnd_pins = [p for p in power_pins if net_util.is_ground_net(p)]
    assert len(vdd_pins) == 1, "Expected to find one VDD pin but found: {}".format(vdd_pins)
    assert len(gnd_pins) == 1, "Expected to find one GND pin but found: {}".format(gnd_pins)
    vdd_pin = vdd_pins[0]
    gnd_pin = gnd_pins[0]

    # Get information on pins
    input_pins, output_pins, output_functions_user = liberty_util.get_pin_information(cell_group)

    if len(input_pins) == 0:
        msg = "Cell has no input pins."
        logger.error(msg)
        assert False, msg

    if len(output_pins) == 0:
        msg = "Cell has no output pins."
        logger.error(msg)
        assert False, msg

    def _transistors2multigraph(transistors) -> nx.MultiGraph:
        """ Create a graph representing the transistor network.
            Each edge corresponds to a transistor, each node to a net.
        """
        G = nx.MultiGraph()
        for t in transistors:
            G.add_edge(t.left, t.right, (t.gate, t.channel_type))
        assert nx.is_connected(G)
        return G

    # Derive boolean functions for the outputs from the netlist.
    logger.debug("Derive boolean functions for the outputs based on the netlist.")
    transistor_graph = _transistors2multigraph(transistors_abstract)
    output_functions_deduced = functional_abstraction.analyze_circuit_graph(graph=transistor_graph,
                                                                            pins_of_interest=io_pins,
                                                                            vdd_pin=vdd_pin,
                                                                            gnd_pin=gnd_pin,
                                                                            user_input_nets=input_pins)
    # Convert keys into strings (they are `sympy.Symbol`s now)
    output_functions_deduced = {output.name: function for output, function in output_functions_deduced.items()}
    output_functions_symbolic = output_functions_deduced

    # Log deduced output functions.
    for output_name, function in output_functions_deduced.items():
        logger.info("Deduced output function: {} = {}".format(output_name, function))

    # Merge deduced output functions with the ones read from the liberty file and perform consistency check.
    for output_name, function in output_functions_user.items():
        logger.info("User supplied output function: {} = {}".format(output_name, function))
        assert output_name in output_functions_deduced, "No function has been deduced for output pin '{}'.".format(
            output_name)
        # Consistency check: verify that the deduced output formula is equal to the one defined in the liberty file.
        equal = functional_abstraction.bool_equals(function, output_functions_deduced[output_name])
        if not equal:
            msg = "User supplied function does not match the deduced function for pin '{}'".format(output_name)
            logger.error(msg)

        if equal:
            # Take the function defined by the liberty file.
            # This might be desired because it is in another form (CND, DNF,...).
            output_functions_symbolic[output_name] = function

    # Convert deduced output functions into Python lambda functions.
    output_functions = {
        name: _boolean_to_lambda(f)
        for name, f in output_functions_symbolic.items()
    }

    # Get timing corner from liberty file.
    # TODO: let user overwrite it.
    default_operating_conditions = library['default_operating_conditions']
    timing_corners = {
        'typical': TimingCorner.TYPICAL,
        'worst': TimingCorner.WORST,
        'best': TimingCorner.BEST,
    }

    assert default_operating_conditions in timing_corners, "Unknown operating condition corner: {}".format(
        default_operating_conditions)

    timing_corner = timing_corners[default_operating_conditions]
    logger.info("Timing corner: {}".format(timing_corner.name))

    # Read trip points from liberty file.
    trip_points = read_trip_points_from_liberty(library)

    logger.debug(trip_points)

    spice_includes = args.include if args.include else []
    if len(spice_includes) == 0:
        logger.warning("No transistor model supplied. Use --include or -I.")

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
            timing_corner=timing_corner,
            spice_netlist_file=args.spice,
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

        # Insert boolean function of output.
        output_pin_group.set_boolean_function('function', output_functions_symbolic[output_pin])

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
                timing_corner=timing_corner,
                spice_netlist_file=args.spice,
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
