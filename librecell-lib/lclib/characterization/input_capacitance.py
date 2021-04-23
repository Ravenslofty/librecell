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
Measurement of the input capacitance by driving the input pin with a constant current.
"""

import os
from typing import List, Optional

from itertools import product

from .util import *
from .piece_wise_linear import *
from .ngspice_subprocess import run_simulation
from lccommon.net_util import get_subcircuit_ports
import tempfile
import logging

from scipy import interpolate

logger = logging.getLogger(__name__)


def characterize_input_capacitances(cell_name: str,
                                    input_pins: List[str],
                                    active_pin: str,
                                    output_pins: List[str],
                                    supply_voltage: float,
                                    trip_points: TripPoints,
                                    timing_corner: CalcMode,
                                    spice_netlist_file: str,
                                    setup_statements: List[str] = None,
                                    time_resolution: float = 1e-12,
                                    temperature=27,
                                    workingdir: Optional[str] = None,
                                    ground_net: str = 'GND',
                                    supply_net: str = 'VDD',
                                    complementary_pins: Optional[Dict[str, str]] = None,
                                    debug: bool = False
                                    ):
    """
    Estimate the input capacitance of the `active_pin`.
    The estimation is done by simulating a constant current flowing into an input and measuring the
    time it takes for the input to go from high to low or low to high. This time multiplied by the current
    yields the transported charge which together with the voltage difference tells the capacitance.
    The measurement is done for all combinations of static inputs (all other inputs that are not measured).

    :param cell_name: Name of the cell to be measured. This must match with the names used in the netlist and liberty file.
    :param input_pins: List of all input pin names.
    :param active_pin: Name of the pin to be measured.
    :param output_pins: List of cell output pins.
    :param supply_voltage: VDD.
    :param trip_points: Trip-point object which specifies the voltage thresholds of the logical values.
    :param timing_corner: Specify whether to take the maximum, minimum or average capacitance value. (Over all static input combinations).
    :param spice_netlist_file: The file containing the netlist of this cell.
    :param setup_statements: SPICE statements that are included at the beginning of the simulation.
        This should be used for .INCLUDE and .LIB statements.
    :param time_resolution: Time resolution of the simulation.
    :param temperature: Temperature of the simulated circuit.
    :param workingdir: Directory where the simulation files will be put. If not specified a temporary directory will be created.
    :param ground_net: The name of the ground net.
    :param supply_net: The name of the supply net.
    :param complementary_pins: Name mapping of differential input pairs. Dict[non inverting pin, inverting pin].
    :param debug: Enable more verbose debugging output such as plots of the simulations.
    """

    # Create temporary working directory.
    if workingdir is None:
        workingdir = tempfile.mkdtemp("lctime-")

    if complementary_pins is None:
        complementary_pins = dict()
    inputs_inverted = complementary_pins.values()
    assert active_pin not in inputs_inverted, f"Active pin '{active_pin}' must not be an inverted pin of a differential pair."
    input_pins_non_inverted = [p for p in input_pins if p not in inputs_inverted]
    active_pin_inverted = complementary_pins.get(active_pin)

    logger.debug("characterize_input_capacitances()")
    # Find ports of the SPICE netlist.
    ports = get_subcircuit_ports(spice_netlist_file, cell_name)
    logger.debug("Subcircuit ports: {}".format(", ".join(ports)))

    logger.debug("Ground net: {}".format(ground_net))
    logger.debug("Supply net: {}".format(supply_net))

    vdd = supply_voltage
    logger.debug("Vdd: {} V".format(vdd))

    # Create a list of include files.
    if setup_statements is None:
        setup_statements = []
    setup_statements = setup_statements + [f".include {spice_netlist_file}"]

    # Load include files.
    for setup in setup_statements:
        logger.debug("Setup statement: ", setup)
    setup_statements_string = "\n".join(setup_statements)

    # Add output load capacitance. Right now this is 0F.
    output_load_statements = "\n".join((f"Cload_{p} {p} GND 0" for p in output_pins))

    # Choose a maximum time to run the simulation.
    time_max = time_resolution * 1e6

    # Find function to summarize different timing arcs.
    reduction_function = {
        CalcMode.WORST: max,
        CalcMode.BEST: min,
        CalcMode.TYPICAL: np.mean
    }[timing_corner]
    logger.info("Reduction function for summarizing multiple timing arcs: {}".format(reduction_function.__name__))

    logger.debug("Measuring input capacitance.")

    # Generate all possible input combinations for the static input pins.
    static_input_nets = [i for i in input_pins_non_inverted if i != active_pin]
    num_inputs = len(static_input_nets)

    static_inputs = list(product(*([[0, 1]] * num_inputs)))
    logger.info(f"Number of static input combinations: {len(static_inputs)}")

    # TODO: How to choose input current?
    input_current = 10000e-9  # A
    logger.info("Input current: {}".format(input_current))

    # Loop through all combinations of inputs.
    capacitances_rising = []
    capacitances_falling = []
    for static_input in static_inputs:
        for input_rising in [True, False]:

            # Get voltages at static inputs.
            input_voltages = {net: supply_voltage * value for net, value in zip(static_input_nets, static_input)}

            # Add input voltages for inverted inputs of differential pairs.
            for p in static_input_nets:
                inv = complementary_pins.get(p)
                if inv is not None:
                    assert inv not in input_voltages
                    # Add the inverted input voltage.
                    input_voltages[inv] = supply_voltage - input_voltages[p]

            logger.debug("Static input voltages: {}".format(input_voltages))

            # Simulation script file path.
            file_name = f"lctime_input_capacitance_" \
                        f"{''.join((f'{net}={v}' for net, v in input_voltages.items()))}_" \
                        f"{'rising' if input_rising else 'falling'}"
            sim_file = os.path.join(workingdir, f"{file_name}.sp")

            # Output file for simulation results.
            sim_output_file = os.path.join(workingdir, f"{file_name}_output.txt")
            # File for debug plot of the waveforms.
            sim_plot_file = os.path.join(workingdir, f"{file_name}_plot.svg")

            # Switch polarity of current for falling edges.
            _input_current = input_current if input_rising else -input_current

            # Get initial voltage of active pin.
            initial_voltage = 0 if input_rising else vdd
            initial_voltage_inv = vdd - initial_voltage

            # Get the breakpoint condition.
            if input_rising:
                breakpoint_statement = f"stop when v({active_pin}) > {vdd * 0.9}"
            else:
                breakpoint_statement = f"stop when v({active_pin}) < {vdd * 0.1}"

            static_supply_voltage_statements = "\n".join(
                (f"Vinput_{net} {ground_net} {net} {voltage}" for net, voltage in input_voltages.items()))

            # Initial node voltages.
            initial_conditions = {
                active_pin: initial_voltage,
                supply_net: supply_voltage
            }
            # Add static input voltages
            initial_conditions.update(input_voltages)
            # Add initial voltage of inverted input pin (if any).
            if active_pin_inverted:
                initial_conditions[active_pin_inverted] = initial_voltage_inv

            # Create SPICE statements for the input current sources that drive the active pin.
            input_current_source_statements = [
                f"Iinput {ground_net} {active_pin} PULSE(0 {_input_current} 1ns 10ps 0ps 100s)"
            ]
            if active_pin_inverted is not None:
                input_current_source_statements.append(
                    f"Iinput_inv {ground_net} {active_pin_inverted} PULSE(0 {-_input_current} 1ns 10ps 0ps 100s)"
                )
            input_current_source_statements = "\n".join(input_current_source_statements)

            # Create ngspice simulation script.
            sim_netlist = f"""* librecell {__name__}
.title Measure input capacitance of pin {active_pin}

.option TEMP={temperature}

{setup_statements_string}

Xcircuit_under_test {" ".join(ports)} {cell_name}

{output_load_statements}

Vsupply {supply_net} {ground_net} {supply_voltage}

* Input current sources.
{input_current_source_statements}

* Static input voltages.
{static_supply_voltage_statements}

* Initial conditions.
* Also all voltages of DC sources must be here if they are needed to compute the initial conditions.
.ic {" ".join((f"v({net})={v}" for net, v in initial_conditions.items()))}

.control

set filetype=ascii
set wr_vecnames

* Breakpoints
{breakpoint_statement}

* Transient simulation, use initial conditions.
tran {time_resolution} {time_max} uic
wrdata {sim_output_file} v({active_pin}) {" ".join((f"v({p})" for p in output_pins))}
exit
.endc

.end
"""

            # Dump netlist.
            logger.debug(sim_netlist)

            # Dump simulation script to the file.
            logger.info(f"Write simulation netlist: {sim_file}")
            open(sim_file, "w").write(sim_netlist)

            # Run simulation.
            logger.debug("Run simulation.")
            stdout, stderr = run_simulation(sim_file)

            # Fetch simulation results.
            logger.debug("Load simulation output.")
            sim_data = np.loadtxt(sim_output_file, skiprows=1)

            if sim_data.ndim != 2:
                logger.error("Simulation failed. No data was written to the output file.")
                if debug:
                    logger.error(f"ngspice: {stderr}")
                assert False, "Simulation failed. No data was written to the output file."

            time = sim_data[:, 0]
            input_voltage = sim_data[:, 1]

            if debug:
                logger.debug("Create plot of waveforms: {}".format(sim_plot_file))
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.close()
                plt.title(f"Measure input capacitance of pin {active_pin}.")
                plt.plot(time, input_voltage, label=active_pin)
                plt.legend()
                plt.savefig(sim_plot_file)
                plt.close()

            # Calculate average derivative of voltage by finding the slope of the line
            # through the crossing point of the voltage with the two thresholds.
            #
            # TODO: How to chose the thresholds?
            if input_rising:
                thresh1 = vdd * trip_points.slew_lower_threshold_rise
                thresh2 = vdd * trip_points.slew_upper_threshold_rise
                assert thresh1 < thresh2
            else:
                thresh1 = vdd * trip_points.slew_upper_threshold_fall
                thresh2 = vdd * trip_points.slew_lower_threshold_fall
                assert thresh1 > thresh2

            # Find transition times for both thresholds.
            transition_time1 = transition_time(input_voltage, time, threshold=thresh1, n=-1)
            transition_time2 = transition_time(input_voltage, time, threshold=thresh2, n=-1)
            assert transition_time2 > transition_time1

            # Compute deltas of time and voltage between the crossing of the two thresholds.
            f_input_voltage = interpolate.interp1d(x=time, y=input_voltage)
            dt = transition_time2 - transition_time1
            dv = f_input_voltage(transition_time2) - f_input_voltage(transition_time1)
            # dv = input_voltage[-1] - input_voltage[0]
            # dt = time[-1] - time[0]

            # Compute capacitance.
            capacitance = float(_input_current) / (float(dv) / float(dt))

            logger.debug("dV: {}".format(dv))
            logger.debug("dt: {}".format(dt))
            logger.debug("I: {}".format(input_current))
            logger.info("Input capacitance {}: {} F".format(active_pin, capacitance))

            if input_rising:
                capacitances_rising.append(capacitance)
            else:
                capacitances_falling.append(capacitance)

    logger.debug("Characterizing input capacitances: Done")

    # Find max, min or average depending on 'reduction_function'.
    logger.debug(
        "Convert capacitances of all timing arcs into the default capacitance ({})".format(reduction_function.__name__))
    final_capacitance_falling = reduction_function(capacitances_falling)
    final_capacitance_rising = reduction_function(capacitances_rising)
    final_capacitance = reduction_function([final_capacitance_falling, final_capacitance_rising])

    return {
        'rise_capacitance': final_capacitance_falling,
        'fall_capacitance': final_capacitance_rising,
        'capacitance': final_capacitance
    }
