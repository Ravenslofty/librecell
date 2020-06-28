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
from typing import List, Dict, Callable
import tempfile
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit.SiUnits import Farad, Second
from PySpice.Unit import *

from PySpice.Logging import Logging

from itertools import product, count
import matplotlib.pyplot as plt

from .util import *
from .piece_wise_linear import *
from .ngspice_simulation import piece_wise_linear_voltage_source, simulate_circuit
from .ngspice_subprocess import run_simulation
from lccommon.net_util import get_subcircuit_ports

import logging

pyspice_logger = Logging.setup_logging()
logger = logging.getLogger(__name__)


def measure_comb_cell(circuit: Circuit,
                      inputs_nets: list,
                      active_pin: str,
                      output_net: str,
                      output_functions: Dict[str, Callable],
                      vdd: float,
                      input_rise_time: Second,
                      input_fall_time: Second,
                      trip_points: TripPoints,
                      temperature: float = 25,
                      output_load_capacitance: Farad = 0.0 @ u_pF,
                      time_step: Second = 100 @ u_ps,
                      simulation_duration_hint: Second = 200 @ u_ns,
                      reduction_function=max
                      ):
    """ Get timing information of combinatorial circuit.

    :param circuit: Circuit to be characterized. (Without output load)
    :param inputs_nets: Names of input signals.
    :param active_pin: Name of the input signal to be toggled.
    :param output_net: Name of the output signal.
    :param output_functions: A dict mapping output pin names to corresponding boolean functions.
    :param vdd: Supply voltage.
    :param input_rise_time:
    :param input_fall_time:
    :param trip_points: TripPoint object containing thresholds.
    :param temperature:
    :param output_load_capacitance:
    :param time_step: Simulation time step.
    :param simulation_duration_hint: A hint on how long to simulate the circuit.
        This should be in the order of magnitude of propagation delays.
        When chosen too short, the simulation time will be prolonged automatically.
    :param reduction_function: Function used to create default timing arc from conditional timing arcs.
        Should be one of {min, max, np.mean}
    :return:
    """
    # Create an independent copy of the circuit.
    logger.debug("Create an independent copy of the circuit.")
    circuit = circuit.clone(title='Timing simulation for pin "{}"'.format(active_pin))

    if float(output_load_capacitance) > 0:
        # Add output capacitance.
        circuit.C('load', circuit.gnd, output_net, output_load_capacitance)

    # Get all input nets that are not toggled during a simulation run.
    logger.info("Get all input nets that are not toggled during a simulation run.")
    static_input_nets = [i for i in inputs_nets if i != active_pin]

    # Get a list of all input combinations that will be used for measuring conditional timing arcs.
    num_inputs = len(static_input_nets)
    static_inputs = list(product(*([[0, 1]] * num_inputs)))

    rise_transition_durations = []
    fall_transition_durations = []
    rise_delays = []
    fall_delays = []

    rise_powers = []
    fall_powers = []

    # Determine length of simulation.
    period = max(simulation_duration_hint, input_rise_time + input_fall_time)
    logger.debug('Length of simulation: {}'.format(period))

    def _is_signal_stable(signal: np.ndarray, samples_per_period: int, sample_point: float = 1.0,
                          epsilon: float = 0.01):
        """ Check if the signal is clearly on HIGH or LOW level right before changing an input.
        :param signal:
        :param sample_point: Where to sample the signal relative to the period.
        :param epsilon: Tolerance. Deviations by epsilon from VDD or GND are still considered as a stable signal.
        :return:
        """
        assert 0 < sample_point <= 1.0
        for idx in range(int(samples_per_period * sample_point) - 1, len(signal), samples_per_period):
            v_norm = signal[idx] / vdd
            stable = abs(v_norm) <= epsilon or abs(1 - v_norm) <= epsilon
            if not stable:
                return False
        return True

    assert output_net in output_functions, \
        "Boolean function not defined for output pin '{}'".format(output_net)
    output_function = output_functions[output_net]

    # Tolerance used for defining a stable signal.
    epsilon = 0.01

    # Loop through all combinations of inputs.
    logger.debug("Loop through all combinations of inputs.")
    for static_input in static_inputs:

        # Check if the output is controllable with this static input.
        bool_inputs = {net: value > 0 for net, value in zip(static_input_nets, static_input)}

        bool_inputs[active_pin] = False
        output_when_false = output_function(**bool_inputs)
        bool_inputs[active_pin] = True
        output_when_true = output_function(**bool_inputs)

        if output_when_false == output_when_true:
            # The output will not change when this input is changed.
            # Simulation of this conditional arc can be skipped.
            logger.debug("Simulation skipped for conditional arc: {}".format(bool_inputs))
            continue

        for input_rising in [True, False]:
            _circuit = circuit.clone(title='Timing simulation for pin "{}"'.format(active_pin))

            bitsequence = [0, 1] if input_rising else [1, 0]

            # Get voltages at static inputs.
            input_voltages = {net: vdd * value @ u_V for net, value in zip(static_input_nets, static_input)}

            logger.debug("Voltages at static inputs: {}".format(input_voltages))

            # Variable for simulation results.
            analysis = None

            # Run the simulation an check if signals settle to a stable state within simulation time.
            # If the signals don't settle, then increase the simulation time and run the simulation again.
            # TODO: Somehow continuing the simulation would be more efficient (if API allows to).
            for i in count():
                __circuit = _circuit.clone(title='Timing simulation for pin "{}"'.format(active_pin))
                # step = min(time_step * 4, period / 8)
                step = time_step
                samples_per_period = int(period / step)
                logger.debug('Low resolution simulation. Iteration %d', i)

                input_wave = bitsequence_to_piece_wise_linear(bitsequence, float(period),
                                                              rise_time=float(input_rise_time),
                                                              fall_time=float(input_fall_time)
                                                              )

                input_wave.y = input_wave.y * vdd

                # Add a voltage source to the circuit.
                # This applies the changing signal to the input pin.
                piece_wise_linear_voltage_source(__circuit, 'data_in',
                                                 active_pin,
                                                 __circuit.gnd,
                                                 input_wave)

                # Run simulation.
                analysis = simulate_circuit(__circuit, input_voltages, step_time=step @ u_s,
                                            end_time=period * len(bitsequence), temperature=temperature)

                # This signals must be stable at the end of the simulation
                must_be_stable = [analysis[active_pin], analysis[output_net]]

                # ... check if they are all stable.
                all_stable = all(
                    (_is_signal_stable(signal, samples_per_period, epsilon=epsilon)
                     for signal in must_be_stable))

                if all_stable:
                    break
                else:
                    period = period * 2

            assert analysis is not None

            time = np.array(analysis.time)
            assert len(time) > 0
            input_voltage = np.array(analysis[active_pin])
            output_voltage = np.array(analysis[output_net])
            supply_current = np.array(analysis['vpower_vdd'])

            # Plot input and output voltage.
            # import matplotlib.pyplot as plt
            # plt.plot(time, input_voltage)
            # plt.plot(time, output_voltage)
            # plt.show()

            assert _is_signal_stable(input_voltage, samples_per_period, epsilon=epsilon)
            assert _is_signal_stable(output_voltage,
                                     samples_per_period,
                                     epsilon=epsilon), "Output signal not stable. Increase simulation time."

            # Skip first period.
            # We are interested in the output signal after the input signal edge.
            time = time[samples_per_period - 1:] - float(period)
            input_voltage = input_voltage[samples_per_period - 1:]
            output_voltage = output_voltage[samples_per_period - 1:]
            supply_current = supply_current[samples_per_period - 1:]

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
            if output_a != output_b:
                logger.debug('Output switching detected.')

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
            else:
                # If a rising edge does not toggle the output, then the falling edge won't too.
                break

    rise_delay = reduction_function(rise_delays)
    fall_delay = reduction_function(fall_delays)
    rise_transition_duration = reduction_function(rise_transition_durations)
    fall_transition_duration = reduction_function(fall_transition_durations)

    rise_power = reduction_function(rise_powers)
    fall_power = reduction_function(fall_powers)

    # TODO: create suited data type for reporting all timing arcs (for all conditions).

    return {
        'cell_rise': rise_delay,
        'cell_fall': fall_delay,
        'rise_transition': rise_transition_duration,
        'fall_transition': fall_transition_duration,
        'rise_power': rise_power,
        'fall_power': fall_power
    }


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

            # Simulation script file path.
            sim_file = tempfile.mktemp(prefix="lctime_sim_script")

            # Output file for simulation results.
            sim_output_file = tempfile.mktemp(prefix="lctime_sim_output")
            logger.info(f"Simulation output file: {sim_output_file}")

            # Get voltages at static inputs.
            input_voltages = {net: supply_voltage * value for net, value in zip(static_input_nets, static_input)}
            logger.debug("Static input voltages: {}".format(input_voltages))

            for input_rising in [True, False]:

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
                pwl_string = ' '.join((
                    '%es %eV' % (float(time), float(voltage))
                    for time, voltage in zip(input_wave.x, input_wave.y)
                ))
                input_source_statement = f"Vdata_in {related_pin} {ground} PWL({pwl_string})"

                # Get initial voltage of active pin.
                initial_voltage = 0 if input_rising else vdd

                # # Get the breakpoint condition.
                if expected_output:
                    breakpoint_statement = f"stop when v({output_pin}) > {vdd * 0.99}"
                else:
                    breakpoint_statement = f"stop when v({output_pin}) < {vdd * 0.01}"

                static_supply_voltage_statemets = "\n".join(
                    (f"Vinput_{net} {ground} {voltage}" for net, voltage in input_voltages.items()))

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
.ic v({related_pin})={initial_voltage} v({supply})={supply_voltage} {" ".join((f"v({net})={v}" for net, v in input_voltages.items()))} v({output_pin})={initial_output_voltage}

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
                if output_a != output_b:
                    logger.debug('Output switching detected.')

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
                else:
                    # If a rising edge does not toggle the output, then the falling edge won't too.
                    break

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
#
# def characterize_comb_cell_old(cell_name: str,
#                            input_pins: List[str],
#                            output_pin: str,
#                            related_pin: str,
#                            output_functions: Dict[str, Callable],
#                            supply_voltage: float,
#                            trip_points: TripPoints,
#                            timing_corner: CalcMode,
#
#                            spice_netlist_file: str,
#                            spice_include_files: List[str] = None,
#                            time_resolution=50 @ u_ps,
#                            temperature=27,
#                            ) -> Dict[str, np.ndarray]:
#     """
#     Calculate the NDLM timing table of a cell for a given timing arc.
#     :param cell_name: The name of the cell in the SPICE model. Required to find it in the spice netlist.
#     :param input_pins: List of input pins.
#     :param output_pin: The output pin of the timing arc.
#     :param related_pin: The input pin of the timing arc.
#     :param output_functions: A dict mapping output pin names to corresponding boolean functions.
#     :param supply_voltage: Supply voltage for characterization.
#     :param trip_points: TripPoints as defined in the liberty library specification.
#     :param timing_corner: One of TimingCorner.WORST, TimingCorner.BEST or TimingCorner.TYPICAL
#         This defines how the default timing arc is calculated from all the conditional timing arcs.
#         WORST: max
#         BEST: min
#         TYPICAL: np.mean
#     :param spice_netlist_file: Path to SPICE netlist containing the subcircuit of the cell.
#     :param spice_include_files: SICE include files such as transistor models.
#     :param time_resolution: Time step of simulation in Pyspice.Units.Seconds.
#     :param temperature: Simulation temperature in celsius.
#     :return: Returns the NDLM timing tables wrapped in a dict:
#     {'cell_rise': 2d-np.ndarray, 'cell_fall': 2d-np.ndarray, ... }
#     """
#     # Find ports of the SPICE netlist.
#     ports = get_subcircuit_ports(spice_netlist_file, cell_name)
#     logger.info("Subcircuit ports: {}".format(", ".join(ports)))
#
#     # TODO: find correct names for GND/VDD from netlist.
#     ground = 'GND'
#     supply = 'VDD'
#
#     circuit = Circuit('Timing simulation of {}'.format(cell_name), ground=ground)
#
#     if spice_include_files is None:
#         spice_include_files = []
#     spice_include_files = spice_include_files + [spice_netlist_file]
#
#     # Load include files.
#     for inc in spice_include_files:
#         logger.info("Include '{}'".format(inc))
#         circuit.include(inc)
#
#     # Instantiate circuit under test.
#     circuit.X('circuit_unter_test', cell_name, *ports)
#
#     # Power supply.
#     circuit.V('power_vdd', supply, circuit.gnd, supply_voltage @ u_V)
#
#     # Define grid points to be evaluated.
#     # TODO: Pass bounds as parameters & find optimal sampling points.
#     total_output_net_capacitance = np.array([0.1, 0.5, 1.2, 3, 4, 5])
#     input_net_transition = np.array([0.06, 0.24, 0.48, 0.9, 1.2, 1.8])
#
#     # Find function to summarize different timing arcs.
#     reduction_function = {
#         CalcMode.WORST: max,
#         CalcMode.BEST: min,
#         CalcMode.TYPICAL: np.mean
#     }[timing_corner]
#
#     # TODO: Spice ABSTOL, RELTOL, CHARGETOL...
#
#     def f(input_transition_time, output_cap):
#         """
#         Evaluate cell timing at a single input-transition-time/output-capacitance point.
#         :param input_transition_time:
#         :param output_cap:
#         :return:
#         """
#         # TODO: handle multiple output pins in one run.
#         r = measure_comb_cell(circuit,
#                               inputs_nets=input_pins,
#                               active_pin=related_pin,
#                               output_net=output_pin,
#                               output_functions=output_functions,
#                               trip_points=trip_points,
#                               input_rise_time=input_transition_time @ u_ns,
#                               input_fall_time=input_transition_time @ u_ns,
#                               output_load_capacitance=output_cap @ u_pF,
#                               vdd=supply_voltage,
#                               temperature=temperature,
#                               time_step=time_resolution,
#                               reduction_function=reduction_function,
#                               simulation_duration_hint=400 @ u_ps)
#         return (np.array(r['cell_rise']),
#                 np.array(r['cell_fall']),
#                 np.array(r['rise_transition']),
#                 np.array(r['fall_transition']))
#
#     f_vec = np.vectorize(f)
#
#     xx, yy = np.meshgrid(total_output_net_capacitance, input_net_transition)
#
#     # Evaluate timing on the grid.
#     cell_rise, cell_fall, rise_transition, fall_transition = f_vec(xx, yy)
#
#     # Return the tables by liberty naming scheme.
#     return {
#         'total_output_net_capacitance': total_output_net_capacitance,
#         'input_net_transition': input_net_transition,
#         'cell_rise': cell_rise,
#         'cell_fall': cell_fall,
#         'rise_transition': rise_transition,
#         'fall_transition': fall_transition
#     }
