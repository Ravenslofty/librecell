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
from typing import List, Dict
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit.SiUnits import Farad, Second
from PySpice.Unit import *
from PySpice.Logging import Logging

from itertools import product

from .util import *
from .piece_wise_linear import *
from .ngspice_simulation import simulate_circuit
from .ngspice_subprocess import run_simulation
from lccommon.net_util import get_subcircuit_ports
import tempfile
import logging
import matplotlib.pyplot as plt

from scipy import interpolate

pyspice_logger = Logging.setup_logging()
logger = logging.getLogger(__name__)


def characterize_input_capacitances(cell_name: str,
                                    input_pins: List[str],
                                    active_pin: str,
                                    output_pins: List[str],
                                    supply_voltage: float,
                                    trip_points: TripPoints,
                                    timing_corner: CalcMode,

                                    spice_netlist_file: str,
                                    spice_include_files: List[str] = None,
                                    time_resolution=50 @ u_ps,
                                    temperature=27,
                                    ):
    logger.debug("characterize_input_capacitances()")
    # Find ports of the SPICE netlist.
    ports = get_subcircuit_ports(spice_netlist_file, cell_name)
    logger.info("Subcircuit ports: {}".format(", ".join(ports)))

    # TODO: find correct names for GND/VDD from netlist.
    ground = 'GND'
    supply = 'VDD'

    vdd = supply_voltage

    circuit = Circuit('Timing simulation of {}'.format(cell_name), ground=ground)

    if spice_include_files is None:
        spice_include_files = []
    spice_include_files = spice_include_files + [spice_netlist_file]

    # Load include files.
    for inc in spice_include_files:
        logger.info("Include '{}'".format(inc))
    include_statements = "\n".join((f".include {i}" for i in spice_include_files))

    # Add output load capacitance. Right now this is 0F.
    output_load_statements = "\n".join((f"Cload_{p} {p} GND 0" for p in output_pins))

    # TODO: How to choose this?
    time_resolution = '1ps'
    time_max = '1000ps'

    # Find function to summarize different timing arcs.
    reduction_function = {
        CalcMode.WORST: max,
        CalcMode.BEST: min,
        CalcMode.TYPICAL: np.mean
    }[timing_corner]
    logger.info("Reduction function for summarizing multiple timing arcs: {}".format(reduction_function.__name__))

    logger.info("Measuring input capactiance.")

    # Generate all possible input combinations for the static input pins.
    static_input_nets = [i for i in input_pins if i != active_pin]
    num_inputs = len(static_input_nets)
    static_inputs = list(product(*([[0, 1]] * num_inputs)))

    # TODO: How to choose input current?
    input_current = 10000e-9  # A
    logger.info("Input current: {}".format(input_current))

    # Loop through all combinations of inputs.
    capacitances_rising = []
    capacitances_falling = []
    for static_input in static_inputs:
        for input_rising in [True, False]:
            # Simulation script file path.
            sim_file = tempfile.mktemp(prefix="lctime_sim_script")

            # Output file for simulation results.
            sim_output_file = tempfile.mktemp(prefix="lctime_sim_output")

            # Get voltages at static inputs.
            input_voltages = {net: supply_voltage * value for net, value in zip(static_input_nets, static_input)}
            logger.debug("Static input voltages: {}".format(input_voltages))

            # Switch polarity of current for falling edges.
            _input_current = input_current if input_rising else -input_current

            # Get initial voltage of active pin.
            initial_voltage = 0 if input_rising else vdd
            # Get the target voltage of the active pin.
            target_voltage = vdd - initial_voltage

            # Get the breakpoint condition.
            if input_rising:
                breakpoint_statement = f"stop when v({active_pin}) > {vdd*0.9}"
            else:
                breakpoint_statement = f"stop when v({active_pin}) < {vdd*0.1}"

            static_supply_voltage_statemets = "\n".join(
                (f"Vinput_{net} {ground} {voltage}" for net, voltage in input_voltages.items()))

            # Create ngspice simulation script.
            sim_netlist = f"""* librecell {__name__}
.title Measure input capacitance of pin {active_pin}

{include_statements}

Xcircuit_under_test {" ".join(ports)} {cell_name}

{output_load_statements}

Vsupply {supply} {ground} {supply_voltage}
Iinput {ground} {active_pin} {_input_current}

* Static input voltages.
{static_supply_voltage_statemets}

* Initial conditions.
* Also all voltages of DC sources must be here if they are needed to compute the initial conditions.
.ic v({active_pin})={initial_voltage} v({supply}={supply_voltage}

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

            print(sim_netlist)
            # Dump simulation script to the file.
            logger.info(f"Write simulation netlist: {sim_file}")
            open(sim_file, "w").write(sim_netlist)

            run_simulation(sim_file)

            logger.debug("Load simulation output.")
            sim_data = np.loadtxt(sim_output_file, skiprows=1)
            time = sim_data[:, 0]
            input_voltage = sim_data[:, 1]

            plt.plot(time, input_voltage)
            plt.show()

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
            transition_time1 = transition_time(input_voltage, time, threshold=thresh1, assert_one_crossing=True)
            transition_time2 = transition_time(input_voltage, time, threshold=thresh2, assert_one_crossing=True)
            assert transition_time2 > transition_time1

            f_input_voltage = interpolate.interp1d(x=time, y=input_voltage)
            dt = transition_time2 - transition_time1
            dv = f_input_voltage(transition_time2) - f_input_voltage(transition_time1)
            # dv = input_voltage[-1] - input_voltage[0]
            # dt = time[-1] - time[0]

            # Compute capacitance
            capacitance = float(_input_current) / (float(dv) / float(dt))

            logger.debug("dV: {}".format(dv))
            logger.debug("dt: {}".format(dt))
            logger.debug("I: {}".format(input_current))
            logger.info("Capacitance: {}".format(capacitance))

            if input_rising:
                capacitances_rising.append(capacitance)
            else:
                capacitances_falling.append(capacitance)

    logger.info("Characterizing input capacitances: Done")

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


# def measure_input_capacitance(circuit: Circuit,
#                               inputs_nets: List[str],
#                               active_pin: str,
#                               output_nets: List[str],
#                               vdd: float,
#                               trip_points: TripPoints,
#                               temperature: float = 25,
#                               output_load_capacitance: Farad = 0.0 @ u_pF,
#                               time_step: Second = 100 @ u_ps,
#                               simulation_duration_hint: Second = 200 @ u_ns,
#                               reduction_function=max
#                               ) -> Dict[str, float]:
#     """ Measure the input capacitance of the `active_pin`.
#
#     :param circuit: Circuit to be characterized. (Without output load)
#     :param inputs_nets: Names of input signals.
#     :param active_pin: Name of the input signal to be toggled.
#     :param output_net: Name of the output signal.
#     :param vdd: Supply voltage.
#     :param trip_points: TripPoints object.
#     :param temperature:
#     :param output_load_capacitance:
#     :param time_step: Simulation time step.
#     :param simulation_duration_hint: A hint on how long to simulate the circuit.
#         This should be in the order of magnitude of propagation delays.
#         When chosen too short, the simulation time will be prolonged automatically.
#     :param reduction_function: Function used to create default timing arc from conditional timing arcs.
#         Should be one of {min, max, np.mean}
#     :return: A dict containing values of 'rise_capacitance' and 'fall_capacitance' in Farads.
#     """
#     logger.debug("measure_input_capacitance()")
#     # Create an independent copy of the circuit.
#     logger.debug("Create an independent copy of the circuit.")
#     circuit = circuit.clone(title='Input capacitance measurement for pin "{}"'.format(active_pin))
#
#     if float(output_load_capacitance) > 0:
#         # Add output capacitances.
#         for output_net in output_nets:
#             circuit.C('load', circuit.gnd, output_net, output_load_capacitance)
#
#     static_input_nets = [i for i in inputs_nets if i != active_pin]
#     num_inputs = len(static_input_nets)
#     static_inputs = list(product(*([[0, 1]] * num_inputs)))
#
#     # TODO: set initial voltage at active_pin.
#
#     input_current = 10000 @ u_nA
#     logger.info("Input current: {}".format(input_current))
#
#     time_step = 1 @ u_ps
#     # Guess of necessary simulation duration.
#     period = 1000 @ u_ps
#     logger.info("Guess of necessary simulation duration: {}".format(period))
#     # Loop through all combinations of inputs.
#     capacitances_rising = []
#     capacitances_falling = []
#     for static_input in static_inputs:
#         for input_rising in [True, False]:
#             _circuit = circuit.clone()
#
#             # Get voltages at static inputs.
#             input_voltages = {net: vdd * value @ u_V for net, value in zip(static_input_nets, static_input)}
#             logger.debug("Static input voltages: {}".format(input_voltages))
#
#             # Switch polarity of current for falling edges.
#             _input_current = input_current if input_rising else -input_current
#             # Create constant current source to drive the active pin.
#             _circuit.I('src_{}'.format(active_pin), circuit.gnd, active_pin, dc_value=_input_current)
#
#             # Get initial voltage of active pin.
#             initial_voltage = 0 @ u_V if input_rising else vdd @ u_V
#
#             # Run simulation
#             # Loop because it might be necessary to run a longer simulation.
#             logger.info("Run simulation.")
#             while True:
#                 analysis = simulate_circuit(_circuit, input_voltages, step_time=time_step,
#                                             end_time=period, temperature=temperature,
#                                             initial_voltages={active_pin: initial_voltage @ u_V}
#                                             )
#
#                 time = np.array(analysis.time)
#                 assert len(time) > 0
#                 input_voltage = np.array(analysis[active_pin])
#                 output_voltage = np.array(analysis[output_nets[0]])
#                 logger.debug("Input voltage at start input_voltage[0]: {}".format(input_voltage[0]))
#                 logger.debug("Input voltage at end input_voltage[-1]: {}".format(input_voltage[-1]))
#
#                 if input_voltage[0] < 0.1 * vdd and input_voltage[-1] > vdd or \
#                         input_voltage[0] > 0.9 * vdd and input_voltage[-1] < 0:
#                     # The input voltage spans the whole range from 0 to vdd.
#                     # So the simulation was long enough.
#                     break
#                 else:
#                     # Simulation was not long enough, double it.
#                     period = period * 2
#                     logger.info("Simulation was not long enough. New simulation time: {}".format(period))
#
#                 if period > 100000 @ u_ps:
#                     logger.info("VDD: {}".format(vdd))
#                     logger.info("input_voltage[0]: {}".format(input_voltage[0]))
#                     logger.info("input_voltage[-1]: {}".format(input_voltage[-1]))
#                     logger.error(
#                         "Error: Simulation took too long! It seems the input voltage did not span the whole range from 0 to vdd!")
#                     break
#
#             # if input_rising:
#             #     input_condition = "B = {}".format(str(input_voltages['B']))
#             #     plt.plot(time*1e9, input_voltage, label='Input when {}'.format(input_condition))
#             #     plt.plot(time*1e9, output_voltage, label='Output when {}'.format(input_condition))
#             # plt.ylabel('input voltage [V]')
#             # plt.xlabel('time [ns]')
#             # plt.xlim(0, 0.3)
#             # plt.legend()
#             # plt.show()
#
#             # Calculate average derivative of voltage by finding the slope of the line
#             # through the crossing point of the voltage with the two thresholds.
#             #
#             # TODO: How to chose the thresholds?
#             if input_rising:
#                 thresh1 = vdd * trip_points.slew_lower_threshold_rise
#                 thresh2 = vdd * trip_points.slew_upper_threshold_rise
#                 assert thresh1 < thresh2
#             else:
#                 thresh1 = vdd * trip_points.slew_upper_threshold_fall
#                 thresh2 = vdd * trip_points.slew_lower_threshold_fall
#                 assert thresh1 > thresh2
#
#             # Find transition times for both thresholds.
#             transition_time1 = transition_time(input_voltage, time, threshold=thresh1, assert_one_crossing=True)
#             transition_time2 = transition_time(input_voltage, time, threshold=thresh2, assert_one_crossing=True)
#             assert transition_time2 > transition_time1
#
#             f_input_voltage = interpolate.interp1d(x=time, y=input_voltage)
#             dt = transition_time2 - transition_time1
#             dv = f_input_voltage(transition_time2) - f_input_voltage(transition_time1)
#             # dv = input_voltage[-1] - input_voltage[0]
#             # dt = time[-1] - time[0]
#
#             # Compute capacitance
#             capacitance = float(_input_current) / (float(dv) / float(dt))
#
#             logger.debug("dV: {}".format(dv))
#             logger.debug("dt: {}".format(dt))
#             logger.debug("I: {}".format(input_current))
#             logger.info("Capacitance: {}".format(capacitance))
#
#             if input_rising:
#                 capacitances_rising.append(capacitance)
#             else:
#                 capacitances_falling.append(capacitance)
#
#     # plt.ylabel('Input voltage [V]')
#     # plt.xlabel('Time [ns]')
#     # plt.ylim(0, vdd*1.1)
#     # plt.xlim(0, 0.25)
#     # plt.legend()
#     # plt.show()
#
#     # Find max, min or average depending on 'reduction_function'.
#     logger.debug(
#         "Convert capacitances of all timing arcs into the default capacitance ({})".format(reduction_function.__name__))
#     final_capacitance_falling = reduction_function(capacitances_falling)
#     final_capacitance_rising = reduction_function(capacitances_rising)
#     final_capacitance = reduction_function([final_capacitance_falling, final_capacitance_rising])
#
#     return {
#         'rise_capacitance': final_capacitance_falling,
#         'fall_capacitance': final_capacitance_rising,
#         'capacitance': final_capacitance
#     }
