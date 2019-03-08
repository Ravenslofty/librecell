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
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Unit.SiUnits import Farad, Second

import PySpice.Logging.Logging as Logging

from itertools import count
from .util import *
from lccommon.net_util import get_subcircuit_ports
from .ngspice_simulation import simulate_circuit
from .piece_wise_linear import *

from scipy import optimize
from math import isclose

logger = Logging.setup_logging()


#
# def _test_plot_flipflop_setup_behavior():
#     trip_points = TripPoints(
#         input_threshold_rise=0.5,
#         input_threshold_fall=0.5,
#         output_threshold_rise=0.5,
#         output_threshold_fall=0.5,
#
#         slew_lower_threshold_rise=0.2,
#         slew_upper_threshold_rise=0.8,
#         slew_lower_threshold_fall=0.2,
#         slew_upper_threshold_fall=0.8
#     )
#
#     subckt_name = 'DFFPOSX1'
#     include_file = '/home/user/FreePDK45/osu_soc/lib/source/netlists/{}.pex.netlist'.format(subckt_name)
#     model_file = '/home/user/FreePDK45/osu_soc/lib/files/gpdk45nm.m'
#
#     ports = get_subcircuit_ports(include_file, subckt_name)
#     print("Ports: ", ports)
#     data_in = 'D'
#     clock = 'CLK'
#     data_out = 'Q'
#     ground = 'GND'
#     supply = 'VDD'
#
#     inputs = [clock, data_in]
#
#     period = 1000 @ u_ps
#     input_rise_time = 60 @ u_ps
#     input_fall_time = 60 @ u_ps
#
#     temperature = 27
#
#     circuit = Circuit('Timing simulation of {}'.format(subckt_name), ground=ground)
#
#     circuit.include(include_file)
#     circuit.include(model_file)
#
#     # Circuit under test.
#     x1 = circuit.X('circuit_unter_test', subckt_name, *ports)
#
#     vdd = 1.1
#
#     # Power supply.
#     circuit.V('power_vdd', supply, circuit.gnd, vdd @ u_V)
#
#     # Output load.
#     C_out = circuit.C('out', circuit.gnd, data_out, 0.1 @ u_pF)
#
#     # Voltage sources for input signals.
#     # input_sources = [circuit.V('in_{}'.format(inp), inp, circuit.gnd, 'dc 0 external') for inp in inputs]
#
#     plt.title('Setup time measurement')
#     plt.xlabel('Time [ps]')
#     plt.ylabel('Voltage [V]')
#     plt.grid()
#
#     is_clk_plotted = False
#     num_plots = 3
#
#     ax_clk = plt.subplot(num_plots, 1, 1)
#     ax_clk.set_title('Clock')
#     ax_clk.set_ylabel("[V]")
#     ax_clk.grid()
#
#     ax_input = plt.subplot(num_plots, 1, 2, sharex=ax_clk)
#     ax_input.set_title('Data Input')
#     ax_input.set_ylabel("[V]")
#     ax_input.grid()
#
#     ax_output = plt.subplot(num_plots, 1, 3, sharex=ax_clk)
#     ax_output.set_title('Data Output')
#     ax_output.set_ylabel("[V]")
#     ax_output.set_xlabel("Time [ps]")
#     ax_output.grid()
#
#     def get_clock_to_output_delay(setup_time: Second,
#                                   hold_time: Second,
#                                   rising_clock_edge: bool,
#                                   rising_data_edge: bool):
#         """ Get the delay from rising clock edge to rising output `Q` edge.
#         :param setup_time: Delay from rising data input `D` edge to rising clock edge.
#         :return:
#         """
#         _circuit = circuit.clone()
#
#         clock_bits = np.array([0, 1, 0, 1, 1])
#         if not rising_clock_edge:
#             clock_bits = 1 - clock_bits
#
#         data_in_bits = np.array([0, 0, 0, 1, 1])
#         if not rising_data_edge:
#             data_in_bits = 1 - data_in_bits
#
#         clk_wave = bitsequence_to_piece_wise_linear_old(clock_bits, float(period),
#                                                         rise_time=float(input_rise_time),
#                                                         fall_time=float(input_fall_time))
#
#         # Create PWL source for input. Shifted by `setup_time`.
#         input_wave = bitsequence_to_piece_wise_linear_old(data_in_bits, float(period),
#                                                           rise_time=float(input_rise_time),
#                                                           fall_time=float(input_fall_time),
#                                                           start_time=-float(setup_time))
#         simulation_end = period * len(clock_bits)
#
#         t_clock_edge = 3 * period
#
#         # Create data pulse.
#         input_wave2 = PulseWave(
#             start_time=float(t_clock_edge - setup_time),
#             duration=float(setup_time + hold_time),
#             polarity=rising_data_edge,
#             rise_time=float(input_rise_time),
#             fall_time=float(input_fall_time),
#             rise_threshold=trip_points.input_threshold_rise,
#             fall_threshold=trip_points.input_threshold_fall
#         )
#         input_wave2 *= vdd
#         input_wave2.add_sampling_point(0)
#         input_wave2.add_sampling_point(float(simulation_end))
#         # input_wave -= PulseWave(start_time=0, duration=1e-10, fall_time=1e-12)
#
#         clk_wave.y = clk_wave.y * vdd
#         input_wave.y = input_wave.y * vdd
#         #
#         # print(input_wave.x)
#         # print(input_wave.y)
#         # print(input_wave2.x)
#         # print(input_wave2.y)
#
#         input_voltages = {
#             clock: clk_wave,
#             data_in: input_wave
#         }
#
#         # Very strange results when using too large time steps
#         time_step = 5 @ u_ps
#         samples_per_period = int(period / time_step)
#         analysis = simulate_circuit(_circuit, input_voltages, time_step=time_step,
#                                     end_time=simulation_end, temperature=temperature)
#
#         time = np.array(analysis.time)
#         clock_voltage = np.array(analysis[clock])
#         input_voltage = np.array(analysis[data_in])
#         output_voltage = np.array(analysis[data_out])
#
#         # plt.plot(time, input_wave(time))
#         # plt.plot(time, input_wave2(time))
#         # plt.show()
#         #
#         if float(setup_time) >= 100e-12:
#             ax_clk.plot(time * 1e12, clock_voltage)
#
#         ax_input.plot(time * 1e12, input_voltage, label='setup time = %0.2f ps' % (setup_time * 1e12))
#         ax_output.plot(time * 1e12, output_voltage, label='setup time = %0.2f ps' % (setup_time * 1e12))
#
#         # Start of interesting interval
#         start = 5 * samples_per_period // 2
#
#         time = time[start:]
#         clock_voltage = clock_voltage[start:]
#         input_voltage = input_voltage[start:]
#         output_voltage = output_voltage[start:]
#
#         if not rising_data_edge:
#             output_voltage = 1 - output_voltage
#
#         # Normalize
#         clock_voltage /= vdd
#         input_voltage /= vdd
#         output_voltage /= vdd
#
#         q0 = output_voltage[0] > 0.5
#         q1 = output_voltage[-1] > 0.5
#
#         if not q0 and q1:
#             # Output has rising edge
#             delay = get_input_to_output_delay(time=time, input_signal=clock_voltage,
#                                               output_signal=output_voltage, trip_points=trip_points)
#         else:
#             delay = float('Inf')
#
#         return delay
#
#     setup_times = np.arange(24e-12, 50e-12, 1e-12)
#     start = 24.4e-12
#     end = 30e-12
#     sp = np.linspace(0, 1, 10) ** 6
#     setup_times = (sp / (sp[-1] - sp[0]) * (end - start)) + start
#     setup_times = np.arange(24e-12, 50e-12, 1e-12)
#     setup_times = [500e-12, 24.42e-12, 0e-12]
#     hold_times = np.arange(00e-12, 100e-12, 10e-12)
#
#     pos_edge_flipflop = True
#
#     delays_01 = np.array(
#         [get_clock_to_output_delay(setup_time=t @ u_s, hold_time=1 @ u_s,
#                                    rising_clock_edge=pos_edge_flipflop, rising_data_edge=True) for t in
#          setup_times])
#
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#     # delays_10 = np.array(
#     #     [get_clock_to_output_delay(setup_time=t @ u_s, hold_time=1 @ u_s,
#     #                                rising_clock_edge=pos_edge_flipflop, rising_data_edge=False) for t in
#     #      setup_times])
#
#     # delays_01 = np.array(
#     #     [get_clock_to_output_delay(setup_time=1 @ u_s, hold_time=t @ u_s,
#     #                                rising_clock_edge=pos_edge_flipflop, rising_data_edge=True) for t in
#     #      hold_times])
#     # delays_10 = np.array(
#     #     [get_clock_to_output_delay(setup_time=1 @ u_s, hold_time=t @ u_s,
#     #                                rising_clock_edge=pos_edge_flipflop, rising_data_edge=False) for t in
#     #      hold_times])
#
#     plt.plot(setup_times * 1e12, delays_01 * 1e12, 'x-', label='D: 0 -> 1')
#     # plt.plot(setup_times * 1e12, delays_10 * 1e12, 'x-', label='D: 1 -> 0')
#     plt.title('Propagation time of {} as function of setup time.'.format(subckt_name))
#     plt.xlabel('setup time (clock arrival time - data arrival time) [ps]')
#     plt.ylabel('propagation time [ps]')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


def get_clock_to_output_delay(
        circuit: Circuit,
        clock_input: str,
        data_in: str,
        data_out: str,
        setup_time: Second,
        hold_time: Second,
        rising_clock_edge: bool,
        rising_data_edge: bool,
        vdd: float,
        input_rise_time: Second,
        input_fall_time: Second,
        trip_points: TripPoints,
        temperature: float = 25,
        output_load_capacitance: Farad = 0.0 @ u_pF,
        time_step: Second = 100 @ u_ps,
        simulation_duration_hint: Second = 200 @ u_ps) -> float:
    """Get the delay from rising clock edge to rising output `Q` edge.

    :param circuit:
    :param clock_input:
    :param data_in:
    :param data_out:
    :param setup_time: Delay from data input `D` edge to rising clock edge.
    :param hold_time: Delay from clock edge to data input edge.
    :param rising_clock_edge:
    :param rising_data_edge:
    :param vdd:
    :param input_rise_time:
    :param input_fall_time:
    :param trip_points:
    :param temperature:
    :param output_load_capacitance:
    :param time_step: Simulation time step.
    Very strange results when using too large time steps
    :param simulation_duration_hint:
    :return:
    """

    _circuit = circuit.clone()

    # Attach output load.
    _circuit.C('out', circuit.gnd, data_out, output_load_capacitance)

    period = max(simulation_duration_hint, input_rise_time + input_fall_time)

    clock_pulse1 = PulseWave(
        start_time=float(period),
        duration=float(period),
        polarity=rising_clock_edge,
        rise_time=float(input_rise_time),
        fall_time=float(input_fall_time),
        rise_threshold=trip_points.input_threshold_rise,
        fall_threshold=trip_points.input_threshold_fall
    )

    t_clock_edge = 4 * period + setup_time

    clock_edge = StepWave(
        start_time=float(t_clock_edge),
        polarity=rising_clock_edge,
        transition_time=float(input_rise_time),
        rise_threshold=trip_points.input_threshold_rise,
        fall_threshold=trip_points.input_threshold_fall
    )
    assert isclose(clock_edge(float(t_clock_edge)),
                   trip_points.input_threshold_rise if rising_clock_edge
                   else trip_points.input_threshold_fall)

    clk_wave = clock_pulse1 + clock_edge

    if not rising_clock_edge:
        # Compensate for offset.
        clk_wave -= 1

    simulation_end = t_clock_edge + 4 * period

    # Create data pulse.
    input_wave = PulseWave(
        start_time=float(t_clock_edge - setup_time),
        duration=float(setup_time + hold_time),
        polarity=rising_data_edge,
        rise_time=float(input_rise_time),
        fall_time=float(input_fall_time),
        rise_threshold=trip_points.input_threshold_rise,
        fall_threshold=trip_points.input_threshold_fall
    )

    input_wave *= vdd
    clk_wave *= vdd

    input_voltages = {
        clock_input: clk_wave,
        data_in: input_wave
    }

    samples_per_period = int(period / time_step)
    analysis = simulate_circuit(_circuit, input_voltages, time_step=time_step,
                                end_time=simulation_end, temperature=temperature)

    time = np.array(analysis.time)
    clock_voltage = np.array(analysis[clock_input])
    input_voltage = np.array(analysis[data_in])
    output_voltage = np.array(analysis[data_out])

    # plt.plot(time, clock_voltage)
    # plt.plot(time, input_voltage)
    # plt.plot(time, output_voltage)
    # plt.show()

    # Start of interesting interval
    start = int((t_clock_edge - period / 2) / period * samples_per_period)

    time = time[start:]
    clock_voltage = clock_voltage[start:]
    input_voltage = input_voltage[start:]
    output_voltage = output_voltage[start:]

    if not rising_data_edge:
        output_voltage = 1 - output_voltage

    # Normalize
    clock_voltage /= vdd
    input_voltage /= vdd
    output_voltage /= vdd

    q0 = output_voltage[0] > 0.5
    q1 = output_voltage[-1] > 0.5

    if not q0 and q1:
        # Output has rising edge
        delay = get_input_to_output_delay(time=time, input_signal=clock_voltage,
                                          output_signal=output_voltage, trip_points=trip_points)
    else:
        delay = float('Inf')

    return delay


def test_plot_flipflop_setup_behavior():
    trip_points = TripPoints(
        input_threshold_rise=0.5,
        input_threshold_fall=0.5,
        output_threshold_rise=0.5,
        output_threshold_fall=0.5,

        slew_lower_threshold_rise=0.2,
        slew_upper_threshold_rise=0.8,
        slew_lower_threshold_fall=0.2,
        slew_upper_threshold_fall=0.8
    )

    subckt_name = 'DFFPOSX1'
    include_file = '/home/user/FreePDK45/osu_soc/lib/source/netlists/{}.pex.netlist'.format(subckt_name)
    model_file = '/home/user/FreePDK45/osu_soc/lib/files/gpdk45nm.m'

    ports = get_subcircuit_ports(include_file, subckt_name)
    print("Ports: ", ports)
    data_in = 'D'
    clock = 'CLK'
    data_out = 'Q'
    ground = 'GND'
    supply = 'VDD'

    input_rise_time = 60 @ u_ps
    input_fall_time = 60 @ u_ps

    temperature = 27

    output_load_capacitance = 0.0 @ u_pF
    time_step = 10 @ u_ps

    # TODO: find appropriate simulation_duration_hint
    simulation_duration_hint = 250 @ u_ps

    circuit = Circuit('Timing simulation of {}'.format(subckt_name), ground=ground)

    circuit.include(include_file)
    circuit.include(model_file)

    # Circuit under test.
    x1 = circuit.X('circuit_unter_test', subckt_name, *ports)

    vdd = 1.1

    # Power supply.
    circuit.V('power_vdd', supply, circuit.gnd, vdd @ u_V)

    # Voltage sources for input signals.
    # input_sources = [circuit.V('in_{}'.format(inp), inp, circuit.gnd, 'dc 0 external') for inp in inputs]

    pos_edge_flipflop = True

    def delay_f(
            setup_time: Second,
            hold_time: Second,
            rising_clock_edge: bool,
            rising_data_edge: bool
    ):
        return get_clock_to_output_delay(
            circuit=circuit,
            clock_input=clock,
            data_in=data_in,
            data_out=data_out,
            setup_time=setup_time,
            hold_time=hold_time,
            rising_clock_edge=rising_clock_edge,
            rising_data_edge=rising_data_edge,
            vdd=vdd,
            input_rise_time=input_rise_time,
            input_fall_time=input_fall_time,
            trip_points=trip_points,
            temperature=temperature,
            output_load_capacitance=output_load_capacitance,
            time_step=time_step,
            simulation_duration_hint=simulation_duration_hint)

    def find_min_data_delay(rising_data_edge: bool) -> Tuple[float, Tuple[PeriodValue, PeriodValue]]:
        """ Find minimum clock->data delay (with large setup/hold window).
        """
        setup_time_guess = input_rise_time
        hold_time_guess = input_fall_time

        setup_time = setup_time_guess
        hold_time = hold_time_guess

        prev_delay = None
        ctr = count()
        for _ in ctr:
            delay = delay_f(setup_time, hold_time,
                            rising_clock_edge=pos_edge_flipflop,
                            rising_data_edge=rising_data_edge)

            if prev_delay is not None and delay != float('Inf'):
                diff = abs(delay - prev_delay)
                fraction = diff / delay
                if fraction < 0.001:
                    # close enough
                    break
            setup_time = setup_time * 2
            hold_time = hold_time * 2

            prev_delay = delay

        logger.debug("Minimum clock to data delay: {}. (Iterations = {})".format(delay, next(ctr)))

        # Return the minimum delay and setup/hold times that lead to it.
        # setup/hold times are devided by 2 because the previous values actually lead to a delay that is close enough.
        return delay, (setup_time / 2, hold_time / 2)

    min_rise_delay, (setup_guess_rise, hold_guess_rise) = find_min_data_delay(rising_data_edge=True)
    min_fall_delay, (setup_guess_fall, hold_guess_fall) = find_min_data_delay(rising_data_edge=False)

    # Define how much delay increase is tolerated.
    # Larger values lead to smaller setup/hold window but to increased delay.
    roll_off_factor = 0.01

    # Define flip flop failure: FF fails if delay is larger than max_accepted_{rise,fall}_delay
    max_rise_delay = min_rise_delay * (1 + roll_off_factor)
    max_fall_delay = min_fall_delay * (1 + roll_off_factor)

    def find_min_setup(rising_data_edge: bool,
                       hold_time: float) -> Tuple[float, float]:
        """
        Find minimal setup time given a fixed hold time.
        Set `hold_time` to a very large value to find the independent minimal setup time.
        :param rising_data_edge: True = rising data edge, False = falling data edge.
        :param hold_time: Fixed hold time.
        :return:
        """
        max_delay = max_rise_delay if rising_data_edge else max_fall_delay
        setup_guess = setup_guess_rise if rising_data_edge else setup_guess_fall

        def f(setup_time: float) -> float:
            """
            Function to find zero.
            :param setup_time:
            :return:
            """
            # print('eval f', setup_time)
            delay = delay_f(setup_time=setup_time, hold_time=hold_time,
                            rising_clock_edge=pos_edge_flipflop,
                            rising_data_edge=rising_data_edge)
            return delay - max_delay

        # x = np.linspace(0, float(setup_guess), 100)
        # y = np.array([f(st) for st in x])
        # plt.plot(x, y)
        # plt.show()

        min_setup_time_indep = optimize.bisect(f, 0, float(setup_guess))
        # min_setup_time_uncond = optimize.newton(f, x0=float(setup_guess))

        return min_setup_time_indep, f(min_setup_time_indep) + max_delay

    def find_min_hold(rising_data_edge: bool,
                      setup_time: float) -> Tuple[float, float]:
        """
        Find minimal hold time given a fixed setup time.
        Set `setup_time` to a very large value to find the independent minimal hold time.
        :param rising_data_edge: True = rising data edge, False = falling data edge.
        :param setup_time: Fixed setup time.
        :return:
        """
        max_delay = max_rise_delay if rising_data_edge else max_fall_delay
        hold_guess = hold_guess_rise if rising_data_edge else hold_guess_fall
        setup_guess = setup_guess_rise if rising_data_edge else setup_guess_fall

        def f(hold_time: float) -> float:
            """
            Function to find zero.
            :param hold_time:
            :return:
            """
            # print('eval f', hold_time)
            # hold_time = max(0 @ u_s, hold_time)
            delay = delay_f(setup_time=setup_time,
                            hold_time=hold_time,
                            rising_clock_edge=pos_edge_flipflop,
                            rising_data_edge=rising_data_edge)
            return delay - max_delay

        # x = np.linspace(-float(1.2*hold_guess), float(2*hold_guess), 80)
        # y = np.array([f(st) for st in x])
        # plt.plot(x, y)
        # plt.show()

        min_hold_time_indep = optimize.bisect(f, -float(setup_guess), float(hold_guess))
        # min_setup_time_uncond = optimize.newton(f, x0=float(setup_guess))

        return min_hold_time_indep, f(min_hold_time_indep) + max_delay

    hold_time_guess = max(hold_guess_rise, hold_guess_fall) * 4
    min_setup_time_uncond_rise, min_setup_delay_rise = find_min_setup(rising_data_edge=True,
                                                                      hold_time=hold_time_guess)
    min_setup_time_uncond_fall, min_setup_delay_fall = find_min_setup(rising_data_edge=False,
                                                                      hold_time=hold_time_guess)

    setup_time_guess = max(setup_guess_rise, setup_guess_fall) * 4
    min_hold_time_uncond_rise, min_hold_delay_rise = find_min_hold(rising_data_edge=True,
                                                                   setup_time=setup_time_guess)
    min_hold_time_uncond_fall, min_hold_delay_fall = find_min_hold(rising_data_edge=False,
                                                                   setup_time=setup_time_guess)

    # Find dependent setup time.
    dependent_setup_time_rise, dependent_setup_delay_rise = \
        find_min_setup(rising_data_edge=True,
                       hold_time=min_hold_time_uncond_rise)

    # dependent_setup_time_fall, dependent_setup_delay_fall = \
    #     find_min_setup(rising_data_edge=False,
    #                    hold_time=min_hold_time_uncond_fall)
    #
    # dependent_hold_time_rise, dependent_hold_delay_rise = \
    #     find_min_hold(rising_data_edge=True,
    #                   hold_time=min_setup_time_uncond_rise)
    #
    # dependent_hold_time_fall, dependent_hold_delay_fall = \
    #     find_min_hold(rising_data_edge=False,
    #                   hold_time=min_setup_time_uncond_fall)

    print("min setup: ", min_setup_time_uncond_rise, min_setup_time_uncond_fall)
    print("max delays: ", min_setup_delay_rise, min_setup_delay_fall)

    print("min hold: ", min_hold_time_uncond_rise, min_hold_time_uncond_fall)
    print("min delays: ", min_hold_delay_rise, min_hold_delay_fall)

    print("dep setup:", dependent_setup_time_rise, dependent_setup_time_fall)
    # print("dep setup delay:", dependent_setup_delay_rise, dependent_setup_delay_fall)
    #
    # print("dep hold:", dependent_hold_time_rise, dependent_hold_time_fall)
    # print("dep hold delay:", dependent_hold_delay_rise, dependent_hold_delay_fall)
