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
import os
from typing import List, Dict, Callable, Optional
import tempfile
from itertools import product
import matplotlib.pyplot as plt

from .util import *
from .piece_wise_linear import *
from .ngspice_subprocess import run_simulation
from lccommon.net_util import get_subcircuit_ports

import logging

logger = logging.getLogger(__name__)


def characterize_comb_cell(cell_name: str,
                           input_pins: List[str],
                           output_pin: str,
                           related_pin: str,
                           output_functions: Dict[str, Callable],
                           supply_voltage: float,
                           trip_points: TripPoints,
                           timing_corner: CalcMode,

                           spice_netlist_file: str,
                           spice_include_files: List[str] = None,
                           time_resolution=1e-12,
                           temperature=27,
                           workingdir: Optional[str] = None
                           ) -> Dict[str, np.ndarray]:
    """
    Calculate the NDLM timing table of a cell for a given timing arc.
    :param cell_name: The name of the cell in the SPICE model. Required to find it in the spice netlist.
    :param input_pins: List of input pins.
    :param output_pin: The output pin of the timing arc.
    :param related_pin: The input pin of the timing arc.
    :param output_functions: A dict mapping output pin names to corresponding boolean functions.
    :param supply_voltage: Supply voltage for characterization.
    :param trip_points: TripPoints as defined in the liberty library specification.
    :param timing_corner: One of TimingCorner.WORST, TimingCorner.BEST or TimingCorner.TYPICAL
        This defines how the default timing arc is calculated from all the conditional timing arcs.
        WORST: max
        BEST: min
        TYPICAL: np.mean
    :param spice_netlist_file: Path to SPICE netlist containing the subcircuit of the cell.
    :param spice_include_files: SICE include files such as transistor models.
    :param time_resolution: Time step of simulation in Pyspice.Units.Seconds.
    :param temperature: Simulation temperature in celsius.
    :return: Returns the NDLM timing tables wrapped in a dict:
    {'cell_rise': 2d-np.ndarray, 'cell_fall': 2d-np.ndarray, ... }
    """

    if workingdir is None:
        workingdir = tempfile.mkdtemp("lctime-")

    # Find ports of the SPICE netlist.
    ports = get_subcircuit_ports(spice_netlist_file, cell_name)
    logger.info("Subcircuit ports: {}".format(", ".join(ports)))

    # TODO: find correct names for GND/VDD from netlist.
    ground = 'GND'
    supply = 'VDD'

    vdd = supply_voltage

    # TODO
    # Maximum simulation time.
    time_max = time_resolution * 1e5

    # Define grid points to be evaluated.
    # TODO: Pass bounds as parameters & find optimal sampling points.
    total_output_net_capacitance = np.array([0.1, 0.5, 1.2, 3, 4, 5])
    input_net_transition = np.array([0.06, 0.24, 0.48, 0.9, 1.2, 1.8])

    # Convert into SI units.
    total_output_net_capacitance = total_output_net_capacitance * 1e-12  # pico farads
    input_net_transition = input_net_transition * 1e-9  # nano seconds

    # Find function to summarize different timing arcs.
    reduction_function = {
        CalcMode.WORST: max,
        CalcMode.BEST: min,
        CalcMode.TYPICAL: np.mean
    }[timing_corner]

    # Load include files.
    if spice_include_files is None:
        spice_include_files = []
    spice_include_files = spice_include_files + [spice_netlist_file]

    for inc in spice_include_files:
        logger.info("Include '{}'".format(inc))
    include_statements = "\n".join((f".include {i}" for i in spice_include_files))

    # Get all input nets that are not toggled during a simulation run.
    logger.info("Get all input nets that are not toggled during a simulation run.")
    static_input_nets = [i for i in input_pins if i != related_pin]
    # Get a list of all input combinations that will be used for measuring conditional timing arcs.
    num_inputs = len(static_input_nets)
    static_inputs = list(product(*([[0, 1]] * num_inputs)))

    # Get boolean function of output pin.
    assert output_pin in output_functions, \
        "Boolean function not defined for output pin '{}'".format(output_pin)
    output_function = output_functions[output_pin]

    def f(input_transition_time, output_cap):
        """
        Evaluate cell timing at a single input-transition-time/output-capacitance point.
        :param input_transition_time:
        :param output_cap:
        :return:
        """

        # Empty results.
        rise_transition_durations = []
        fall_transition_durations = []
        rise_delays = []
        fall_delays = []

        rise_powers = []
        fall_powers = []

        for static_input in static_inputs:

            # Check if the output is controllable with this static input.
            bool_inputs = {net: value > 0 for net, value in zip(static_input_nets, static_input)}

            bool_inputs[related_pin] = False
            output_when_false = output_function(**bool_inputs)
            bool_inputs[related_pin] = True
            output_when_true = output_function(**bool_inputs)

            if output_when_false == output_when_true:
                # The output will not change when this input is changed.
                # Simulation of this conditional arc can be skipped.
                logger.debug("Simulation skipped for conditional arc (output does not toggle): {}".format(bool_inputs))
                continue

            # Get voltages at static inputs.
            input_voltages = {net: supply_voltage * value for net, value in zip(static_input_nets, static_input)}
            logger.debug("Static input voltages: {}".format(input_voltages))

            for input_rising in [True, False]:

                # Simulation script file path.
                file_name = f"lctime_combinational_" \
                            f"slew={input_transition_time}_" \
                            f"load={output_cap}" \
                            f"{''.join((f'{net}={v}' for net, v in input_voltages.items()))}_" \
                            f"{'rising' if input_rising else 'falling'}"
                sim_file = os.path.join(workingdir, f"{file_name}.sp")

                # Output file for simulation results.
                sim_output_file = os.path.join(workingdir, f"{file_name}_output.txt")

                bool_inputs[related_pin] = input_rising
                expected_output = output_function(**bool_inputs)
                initial_output_voltage = 0 if expected_output else vdd

                # Get voltages at static inputs.
                input_voltages = {net: vdd * value for net, value in zip(static_input_nets, static_input)}
                logger.debug("Voltages at static inputs: {}".format(input_voltages))

                # Get stimulus signal for related pin.
                input_wave = StepWave(start_time=0, polarity=input_rising,
                                      rise_threshold=0,
                                      fall_threshold=1,
                                      transition_time=input_transition_time)
                input_wave.y = input_wave.y * vdd

                # Create SPICE format of the piece wise linear source.
                input_source_statement = f"Vdata_in {related_pin} {ground} PWL({input_wave.to_spice_pwl_string()})"

                # Get initial voltage of active pin.
                initial_voltage = 0 if input_rising else vdd

                # # Get the breakpoint condition.
                if expected_output:
                    breakpoint_statement = f"stop when v({output_pin}) > {vdd * 0.99}"
                else:
                    breakpoint_statement = f"stop when v({output_pin}) < {vdd * 0.01}"

                static_supply_voltage_statemets = "\n".join(
                    (f"Vinput_{net} {ground} {voltage}" for net, voltage in input_voltages.items()))

                # Initial node voltages.
                initial_conditions = {
                    related_pin: initial_voltage,
                    supply: supply_voltage,
                    output_pin: initial_output_voltage
                }
                initial_conditions.update(input_voltages)

                # Create ngspice simulation script.
                sim_netlist = f"""* librecell {__name__}
.title Timing simulation for pin '{related_pin}', input_rising={input_rising}.

.option TEMP={temperature}

{include_statements}

Xcircuit_under_test {" ".join(ports)} {cell_name}

* Output load.
Cload {output_pin} {ground} {output_cap}

Vsupply {supply} {ground} {supply_voltage}

* Static input voltages.
{static_supply_voltage_statemets}

* Active input signal.
{input_source_statement}

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
wrdata {sim_output_file} i(vsupply) v({related_pin}) v({output_pin})
exit
.endc

.end
"""

                logger.debug(sim_netlist)
                # Dump simulation script to the file.
                logger.info(f"Write simulation netlist: {sim_file}")
                open(sim_file, "w").write(sim_netlist)

                logger.info("Run simulation.")
                run_simulation(sim_file)

                logger.debug("Load simulation output.")
                sim_data = np.loadtxt(sim_output_file, skiprows=1)

                # Retreive data.
                time = sim_data[:, 0]
                supply_current = sim_data[:, 1]
                input_voltage = sim_data[:, 3]
                output_voltage = sim_data[:, 5]

                # plt.title(f"")
                # plt.plot(time, input_voltage, 'x-', label='input voltage')
                # plt.plot(time, output_voltage, label='output voltage')
                # plt.plot(time, supply_current, label='supply current')
                # plt.legend()
                # plt.show()

                # TODO: What unit does rise_power/fall_power have in liberty files???
                # Is it really power or energy?
                switching_power = np.mean(supply_current * vdd)
                switching_energy = switching_power * (time[-1] - time[0])

                v_thresh = 0.5 * vdd

                # Get input signal before switching and after.
                input_a = input_voltage[0] > v_thresh
                input_b = input_voltage[-1] > v_thresh
                assert input_a != input_b

                # Get output signal before switching and after.
                output_a = output_voltage[0] > v_thresh
                output_b = output_voltage[-1] > v_thresh

                # There should be an edge in the output signal.
                # Because the input signals have been chosen that way.
                assert output_a != output_b, "Supplied boolean function and simulation are inconsistent."

                # Check if output signal toggles.
                assert output_a != output_b, "Output did not toggle."

                output_rising = output_b

                # Normalize input/output such that both have a rising edge.
                input_voltage = input_voltage if input_rising else vdd - input_voltage
                output_voltage = output_voltage if output_rising else vdd - output_voltage
                # Normalize to range [0, ..., 1]
                input_voltage /= vdd
                output_voltage /= vdd

                # Check if signals are already stabilized after one `period`.
                assert abs(input_voltage[0]) < 0.01, "Input signal not yet stable at start."
                assert abs(1 - input_voltage[-1]) < 0.01, "Input signal not yet stable at end."
                assert abs(output_voltage[0]) < 0.01, "Output signal not yet stable at start."
                assert abs(1 - output_voltage[-1]) < 0.01, "Output signal not yet stable at end."

                # Calculate the output slew time: the time the output signal takes to change from
                # `slew_lower_threshold` to `slew_upper_threshold`.
                output_transition_duration = get_slew_time(time, output_voltage, trip_points=trip_points)

                # Calculate delay from the moment the input signal crosses `input_threshold` to the moment the output
                # signal crosses `output_threshold`.
                cell_delay = get_input_to_output_delay(time, input_voltage, output_voltage,
                                                       trip_points=trip_points)

                if output_rising:
                    rise_delays.append(cell_delay)
                    rise_transition_durations.append(output_transition_duration)
                else:
                    fall_delays.append(cell_delay)
                    fall_transition_durations.append(output_transition_duration)

                if input_rising:
                    rise_powers.append(switching_energy)
                else:
                    fall_powers.append(switching_energy)

        return (np.array(rise_delays), np.array(fall_delays),
                np.array(rise_transition_durations), np.array(fall_transition_durations))

    f_vec = np.vectorize(f)

    xx, yy = np.meshgrid(input_net_transition, total_output_net_capacitance)

    # Evaluate timing on the grid.
    cell_rise, cell_fall, rise_transition, fall_transition = f_vec(xx, yy)

    # Return the tables by liberty naming scheme.
    return {
        'total_output_net_capacitance': total_output_net_capacitance,
        'input_net_transition': input_net_transition,
        'cell_rise': cell_rise,
        'cell_fall': cell_fall,
        'rise_transition': rise_transition,
        'fall_transition': fall_transition
    }
