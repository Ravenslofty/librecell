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
import tempfile
import matplotlib.pyplot as plt
from itertools import count

from .ngspice_subprocess import run_simulation

from .util import *
from lccommon.net_util import get_subcircuit_ports
from .piece_wise_linear import *

from scipy import optimize
from math import isclose

from typing import List
import logging

logger = logging.getLogger(__name__)


def get_clock_to_output_delay(
        cell_name: str,
        cell_ports: List[str],
        clock_input: str,
        data_in: str,
        data_out: str,
        setup_time: float,
        hold_time: float,
        rising_clock_edge: bool,
        rising_data_edge: bool,
        supply_voltage: float,
        input_rise_time: float,
        input_fall_time: float,
        trip_points: TripPoints,
        temperature: float = 25,
        output_load_capacitance: float = 0.0,
        time_step: float = 100.0e-12,
        simulation_duration_hint: float = 200.0e-12,
        spice_include_files: List[str] = None,
) -> float:
    """Get the delay from rising clock edge to rising output `Q` edge.

    :param circuit:
    :param clock_input:
    :param data_in:
    :param data_out:
    :param setup_time: Delay from data input `D` edge to rising clock edge.
    :param hold_time: Delay from clock edge to data input edge.
    :param rising_clock_edge:
    :param rising_data_edge:
    :param supply_voltage:
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

    # TODO: No hardcoded netnames!
    ground = 'GND'
    supply = 'VDD'

    logger.info("get_clock_to_output_delay() ...")

    # Load include files.
    if spice_include_files is None:
        spice_include_files = []

    for inc in spice_include_files:
        logger.info("Include '{}'".format(inc))
    include_statements = "\n".join((f".include {i}" for i in spice_include_files))

    period = max(simulation_duration_hint, input_rise_time + input_fall_time)

    clock_pulse1 = PulseWave(
        start_time=period,
        duration=period,
        polarity=rising_clock_edge,
        rise_time=input_rise_time,
        fall_time=input_fall_time,
        rise_threshold=trip_points.input_threshold_rise,
        fall_threshold=trip_points.input_threshold_fall
    )

    t_clock_edge = 4 * period + setup_time

    clock_edge = StepWave(
        start_time=t_clock_edge,
        polarity=rising_clock_edge,
        transition_time=input_rise_time,
        rise_threshold=trip_points.input_threshold_rise,
        fall_threshold=trip_points.input_threshold_fall
    )
    assert isclose(clock_edge(t_clock_edge),
                   trip_points.input_threshold_rise if rising_clock_edge
                   else trip_points.input_threshold_fall)

    clk_wave = clock_pulse1 + clock_edge

    if not rising_clock_edge:
        # Compensate for offset.
        clk_wave -= 1

    simulation_end = t_clock_edge + 4 * period

    # Create data pulse.
    logger.debug("Create data pulse.")
    input_wave = PulseWave(
        start_time=t_clock_edge - setup_time,
        duration=setup_time + hold_time,
        polarity=rising_data_edge,
        rise_time=input_rise_time,
        fall_time=input_fall_time,
        rise_threshold=trip_points.input_threshold_rise,
        fall_threshold=trip_points.input_threshold_fall
    )

    input_wave *= supply_voltage
    clk_wave *= supply_voltage

    input_source_statement = f"Vdata_in {data_in} {ground} PWL({input_wave.to_spice_pwl_string()}) DC=0"
    clk_source_statement = f"Vclk {clock_input} {ground} PWL({clk_wave.to_spice_pwl_string()}) DC=0"

    input_voltages = {
        clock_input: clk_wave,
        data_in: input_wave
    }

    samples_per_period = int(period / time_step)
    logger.debug("Run simulation.")

    sim_file = tempfile.mktemp(prefix='lctime-', suffix='.sp', dir='/dev/shm')
    sim_output_file = tempfile.mktemp(prefix='lctime-out-', suffix='.txt', dir='/dev/shm')

    # TODO
    initial_conditions = {
        supply: supply_voltage,
        data_in: input_wave(0),
        clock_input: clk_wave(0),
        data_out: 0 if rising_data_edge else supply_voltage
    }

    # TODO
    if rising_data_edge:
        breakpoint_statement = f"stop when v({data_out}) > {supply_voltage * 0.99}"
    else:
        breakpoint_statement = f"stop when v({data_out}) < {supply_voltage * 0.01}"

    # Create ngspice simulation script.
    sim_netlist = f"""* librecell {__name__}
.title Measure constraint '{data_in}'-'{clock_input}'->'{data_out}', rising_clock_edge={rising_clock_edge}.

.option TEMP={temperature}

{include_statements}

Xcircuit_under_test {" ".join(cell_ports)} {cell_name}

* Output load.
Cload {data_out} {ground} {output_load_capacitance}

Vsupply {supply} {ground} {supply_voltage}

* Static input voltages.
* TODO {{static_supply_voltage_statemets}}

* Active input signals (clock & data_in).
{clk_source_statement}
{input_source_statement}

* Initial conditions.
* Also all voltages of DC sources must be here if they are needed to compute the initial conditions.
.ic {" ".join((f"v({net})={v}" for net, v in initial_conditions.items()))}

.control 
*option reltol=1e-5
*option abstol=1e-15

set filetype=ascii
set wr_vecnames

* Breakpoints
{breakpoint_statement}

* Transient simulation, use initial conditions.
tran {time_step} {simulation_end} uic
wrdata {sim_output_file} i(vsupply) v({data_in}) v({clock_input}) v({data_out})
exit
.endc

.end
"""
    # logger.debug(sim_netlist)

    # Dump simulation script to the file.
    print(f"Write simulation netlist: {sim_file}")
    open(sim_file, "w").write(sim_netlist)

    logger.info("Run simulation.")
    run_simulation(sim_file)

    logger.debug("Load simulation output.")
    sim_data = np.loadtxt(sim_output_file, skiprows=1)

    # os.remove(sim_file)
    # os.remove(sim_output_file)

    # Retreive data.
    time = sim_data[:, 0]
    supply_current = sim_data[:, 1]
    input_voltage = sim_data[:, 3]
    clock_voltage = sim_data[:, 5]
    output_voltage = sim_data[:, 7]

    # plt.plot(time, clock_voltage, label='clk')
    # plt.plot(time, input_voltage, label='din')
    # plt.plot(time, output_voltage, label='dout')
    # plt.legend()
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
    logger.debug("Normalize voltages (divide by VDD).")
    clock_voltage /= supply_voltage
    input_voltage /= supply_voltage
    output_voltage /= supply_voltage

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
    import os
    base = os.path.expanduser("~")
    include_file = f'{base}/FreePDK45/osu_soc/lib/source/netlists/{subckt_name}.pex.netlist'
    model_file = f'{base}/FreePDK45/osu_soc/lib/files/gpdk45nm.m'

    ports = get_subcircuit_ports(include_file, subckt_name)
    print("Ports: ", ports)
    data_in = 'D'
    clock = 'CLK'
    data_out = 'Q'
    ground = 'GND'
    supply = 'VDD'

    input_rise_time = 0.010e-9
    input_fall_time = 0.010e-9

    temperature = 27

    output_load_capacitance = 0.06e-12

    time_step = 10e-12

    # TODO: find appropriate simulation_duration_hint
    simulation_duration_hint = 250e-12

    # SPICE include files.
    includes = [include_file, model_file]

    vdd = 1.1

    # Voltage sources for input signals.
    # input_sources = [circuit.V('in_{}'.format(inp), inp, circuit.gnd, 'dc 0 external') for inp in inputs]

    pos_edge_flipflop = True

    # Cache for faster re-evaluation of `delay_f`
    cache = dict()

    def delay_f(
            setup_time: float,
            hold_time: float,
            rising_clock_edge: bool,
            rising_data_edge: bool
    ):
        """
        Wrapper around `get_clock_to_output_delay()`. Results are cached such that a further call with same arguments returns the
        cached value of the first call.
        :param setup_time:
        :param hold_time:
        :param rising_clock_edge:
        :param rising_data_edge:
        :return:
        """
        print(f"evaluate delay_f({setup_time}, {hold_time}, {rising_clock_edge}, {rising_data_edge})")

        cache_tag = (setup_time, hold_time, rising_clock_edge, rising_data_edge)
        result = cache.get(cache_tag)
        if result is None:
            result = get_clock_to_output_delay(

                cell_name=subckt_name,
                cell_ports=ports,
                clock_input=clock,
                data_in=data_in,
                data_out=data_out,
                setup_time=setup_time,
                hold_time=hold_time,
                rising_clock_edge=rising_clock_edge,
                rising_data_edge=rising_data_edge,
                supply_voltage=vdd,
                input_rise_time=input_rise_time,
                input_fall_time=input_fall_time,
                trip_points=trip_points,
                temperature=temperature,
                output_load_capacitance=output_load_capacitance,
                time_step=time_step,
                simulation_duration_hint=simulation_duration_hint,
                spice_include_files=includes)
            cache[cache_tag] = result
        else:
            print('Cache hit.')
        return result

    def find_min_data_delay(rising_data_edge: bool) -> Tuple[float, Tuple[float, float]]:
        """ Find minimum clock->data delay (with large setup/hold window).
        """
        setup_time_guess = input_rise_time
        hold_time_guess = input_fall_time

        setup_time = setup_time_guess
        hold_time = hold_time_guess

        assert setup_time != 0  # Does not terminate otherwise.
        assert hold_time != 0  # Does not terminate otherwise.

        prev_delay = None
        delay = None
        ctr = count()
        for _ in ctr:
            delay = delay_f(setup_time, hold_time,
                            rising_clock_edge=pos_edge_flipflop,
                            rising_data_edge=rising_data_edge)

            if prev_delay is not None and delay != float('Inf'):
                diff = abs(delay - prev_delay)
                fraction = diff / delay
                if fraction < 0.001:
                    # Close enough.
                    break
            setup_time = setup_time * 2
            hold_time = hold_time * 2

            prev_delay = delay

        logger.info(f"Minimum clock to data delay: {delay}. (Iterations = {next(ctr)})")

        # Return the minimum delay and setup/hold times that lead to it.
        # setup/hold times are devided by 2 because the previous values actually lead to a delay that is close enough.
        return delay, (setup_time / 2, hold_time / 2)

    min_rise_delay, (setup_guess_rise, hold_guess_rise) = find_min_data_delay(rising_data_edge=True)
    min_fall_delay, (setup_guess_fall, hold_guess_fall) = find_min_data_delay(rising_data_edge=False)

    print(f"min_rise_delay = {min_rise_delay}")
    print(f"min_fall_delay = {min_fall_delay}")

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

        logger.info(f"Find min. setup time. Hold time = {hold_time}")

        def f(setup_time: float) -> float:
            """
            Optimization function.
            Find `setup_time` such that the delay equals the maximum allowed delay.
            :param setup_time:
            :return:
            """
            # print('eval f', setup_time)
            # assert setup_time + hold_time >= input_rise_time + input_fall_time
            delay = delay_f(setup_time=setup_time, hold_time=hold_time,
                            rising_clock_edge=pos_edge_flipflop,
                            rising_data_edge=rising_data_edge)
            return delay - max_delay

        # Determine min and max setup time for binary search.
        shortest = -hold_time + input_rise_time + input_fall_time
        longest = setup_guess
        a = f(shortest)
        b = f(longest)
        assert a > 0
        # Make sure f(longest) is larger than zero.
        while not b < 0:
            longest = longest*2
            b = f(longest)

        # x = np.linspace(shortest, longest, 2)
        # y = np.array([f(st) for st in x])
        # print(x)
        # print(y)
        # # plt.title(f'setup time for $t_{{hold}} = {hold_time}$')
        # # plt.xlabel(f'setup time')
        # # plt.plot(x, y)
        # # plt.show()
        # # exit()

        xtol = 1e-20
        min_setup_time_indep = optimize.bisect(f, shortest, longest, xtol=xtol)
        delay = f(min_setup_time_indep)
        # Check if we really found the root of `f`.
        assert np.allclose(0, delay, atol=xtol * 1000), "Failed to find solution for minimal setup time."

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

        def f(hold_time: float) -> float:
            """
            Function to find zero.
            :param hold_time:
            :return:
            """
            # print('eval f', hold_time)
            delay = delay_f(setup_time=setup_time,
                            hold_time=hold_time,
                            rising_clock_edge=pos_edge_flipflop,
                            rising_data_edge=rising_data_edge)
            return delay - max_delay

        # print(hold_guess)
        # x = np.linspace(-1*hold_guess, 1*hold_guess, 100)
        # y = np.array([f(st) for st in x])
        # plt.plot(x, y, 'x-')
        # plt.show()
        # exit(1)
        # Determine min and max hold time for binary search.
        shortest = -setup_time + input_rise_time + input_fall_time
        longest = hold_guess
        a = f(shortest)
        b = f(longest)
        assert a > 0
        # Make sure f(longest) is larger than zero.
        while not b < 0:
            longest = longest*2
            b = f(longest)

        # x = np.linspace(shortest, longest, 2)
        # y = np.array([f(st) + max_delay for st in x])
        # print(x)
        # print(y)
        # # plt.title(f'setup time for $t_{{hold}} = {hold_time}$')
        # # plt.xlabel(f'setup time')
        # # plt.plot(x, y)
        # # plt.show()
        # # exit()

        xtol = 1e-20
        min_hold_time_indep = optimize.bisect(f, shortest, longest, xtol=xtol)
        delay = f(min_hold_time_indep)
        print(delay)
        # Check if we really found the root of `f`.
        assert np.allclose(0, delay, atol=xtol * 1000), "Failed to find solution for minimal hold time."

        return min_hold_time_indep, f(min_hold_time_indep) + max_delay

    print("Measure unconditional minimal setup time.")
    hold_time_guess = max(hold_guess_rise, hold_guess_fall) * 4
    min_setup_time_uncond_rise, min_setup_delay_rise = find_min_setup(rising_data_edge=True,
                                                                      hold_time=hold_time_guess)
    min_setup_time_uncond_fall, min_setup_delay_fall = find_min_setup(rising_data_edge=False,
                                                                      hold_time=hold_time_guess)

    print(f"unconditional min. setup time rise: {min_setup_time_uncond_rise}")
    print(f"unconditional min. setup time fall: {min_setup_time_uncond_fall}")
    print(f"max delays (rise): {min_setup_delay_rise}")
    print(f"max delays (fall): {min_setup_delay_fall}")

    print("Measure unconditional minimal hold time.")
    setup_time_guess = max(setup_guess_rise, setup_guess_fall) * 40
    min_hold_time_uncond_rise, min_hold_delay_rise = find_min_hold(rising_data_edge=True,
                                                                   setup_time=setup_time_guess)
    min_hold_time_uncond_fall, min_hold_delay_fall = find_min_hold(rising_data_edge=False,
                                                                   setup_time=setup_time_guess)

    print(f"unconditional min. hold time rise: {min_hold_time_uncond_rise}")
    print(f"unconditional min. hold time fall: {min_hold_time_uncond_fall}")
    print(f"max delays (rise): {min_hold_delay_rise}")
    print(f"max delays (fall): {min_hold_delay_fall}")


    # min_hold_time_uncond_rise = -3.465480288494276e-11  # TODO remove
    # min_hold_time_uncond_rise = 1e-11  # TODO remove

    # # Find dependent setup time.
    dependent_setup_time_rise, dependent_setup_delay_rise = \
        find_min_setup(rising_data_edge=True,
                       hold_time=min_hold_time_uncond_rise)

    dependent_setup_time_fall, dependent_setup_delay_fall = \
        find_min_setup(rising_data_edge=False,
                       hold_time=min_hold_time_uncond_fall)

    dependent_hold_time_rise, dependent_hold_delay_rise = \
        find_min_hold(rising_data_edge=True,
                      setup_time=min_setup_time_uncond_rise)

    dependent_hold_time_fall, dependent_hold_delay_fall = \
        find_min_hold(rising_data_edge=False,
                      setup_time=min_setup_time_uncond_fall)

    print("dep setup:", dependent_setup_time_rise, dependent_setup_time_fall)
    print("dep setup delay:", dependent_setup_delay_rise, dependent_setup_delay_fall)
    #
    print("dep hold:", dependent_hold_time_rise, dependent_hold_time_fall)
    print("dep hold delay:", dependent_hold_delay_rise, dependent_hold_delay_fall)
