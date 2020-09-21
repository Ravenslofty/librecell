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
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Parser import SpiceParser
from PySpice.Unit import *
from PySpice.Unit.SiUnits import Farad, Second

import PySpice.Logging.Logging as Logging
import numpy as np

from .ngspice_simulation import piece_wise_linear_voltage_source
from .piece_wise_linear import PieceWiseLinear

import matplotlib.pyplot as plt


# def test_simulate_circuit():
#     time_step = 1 @ u_ms
#     end_time = 10 @ u_s
#
#     circuit = Circuit(title='test1')
#
#     # circuit.V('VDD', 'vdd', circuit.gnd, 10 @ u_V)
#     circuit.R('1', 'input', 'a', 1 @ u_kOhm)
#     circuit.C('1', circuit.gnd, 'a', 1000 @ u_uF)
#
#     ngspice_shared = NgSpiceShared.new_instance(send_data=False)
#
#     piece_wise_linear_voltage_source(circuit, 'src', 'input', circuit.gnd,
#                                      PieceWiseLinear([0, 1, 2, 10], [0, 1, 1, 1]))
#
#     # simulator = circuit.simulator(temperature=25,
#     #                               nominal_temperature=25,
#     #                               simulator='ngspice-shared',
#     #                               ngspice_shared=ngspice_shared)
#
#     ngspice_shared.stop('v(a) > 0.5')
#     # ngspice_shared.destroy()
#     # print(str(simulator))
#     # ngspice_shared.load_circuit(str(simulator))
#     # ngspice_shared.run()
#     # analysis = ngspice_shared.plot(simulator, 'asdf').to_analysis()
#
#     simulator = circuit.simulator(temperature=25,
#                                   nominal_temperature=25,
#                                   simulator='ngspice-shared',
#                                   ngspice_shared=ngspice_shared)
#     analysis = simulator.transient(step_time=time_step, end_time=end_time)
#
#     print(np.array(analysis['a']))
#
#     # plt.plot(analysis['a'])
#     # plt.plot(analysis['input'])
#     # plt.show()


# def test_simulate_circuit_lowlevel_control():
#     """
#     Use `NgSpiceShared` to get finer grained control over the simulation.
#     Especially needed for breakpoints.
#     """
#     circuit = Circuit(title='test2')
#
#     circuit.R('1', 'input', 'a', 1 @ u_kOhm)
#     circuit.C('1', circuit.gnd, 'a', 1000 @ u_uF)
#
#     # piece_wise_linear_voltage_source(circuit, 'src', 'input', circuit.gnd,
#     #                                  PieceWiseLinear([0, 1, 2, 10], [0, 1, 1, 0]))
#     circuit.V('src', 'input', circuit.gnd, 0)
#
#     ngs = NgSpiceShared.new_instance(send_data=False)
#
#     ngs.destroy()
#     ngs.remove_circuit()
#
#     netlist = "{}\n.end".format(str(circuit))
#     print(netlist)
#     ngs.load_circuit(netlist)
#     # Change capacitance.
#     # ngs.alter_device('C1', C="1000uF")
#     # Change signal of piece wise linar voltage source..
#     wave = PieceWiseLinear([0, 1, 10], [0, 1, 1])
#     pwl_string = ' '.join((
#         '%es %eV' % (float(time), float(voltage))
#         for time, voltage in zip(wave.x, wave.y)
#     ))
#     ngs.alter_device('Vsrc', PWL="({})".format(pwl_string))
#
#     # Set temperature.
#     ngs.option(temp=25)
#     ngs.option(nomtemp=25)
#     # Set breakpoint.
#     ngs.stop('v(a) > 0.9')
#     # Run simulation.
#     ngs.exec_command("tran 100ms 60s")
#     ngs.run(background=False)
#
#     # Retreive signals.
#     plot_name = ngs.last_plot
#     print(plot_name)
#     if plot_name == 'const':
#         raise Exception("Simulation failed.")
#     plot = ngs.plot(None, plot_name)
#     analysis = plot.to_analysis()
#
#     print(np.array(analysis['a']))
#     plt.plot(analysis.time, analysis['input'], 'x-')
#     plt.plot(analysis.time, np.array(analysis['a']), 'x-')
#     plt.show()


def test3():

    netlist = r""".title Input capacitance measurement for pin "A"
.include /home/kramert/unison/local/blackbird/git/librecell-examples/librecell_invx1_example/gpdk45nm.m
.include /home/kramert/unison/local/blackbird/git/librecell-examples/librecell_invx1_example/INVX1.pex.netlist
Xcircuit_unter_test A GND VDD Y INVX1
Vpower_vdd VDD GND 1.1V
Isrc_A GND A 10000nA

.end

"""
    ngs = NgSpiceShared.new_instance(send_data=False)

    ngs.destroy()
    ngs.remove_circuit()

    print(netlist)
    ngs.load_circuit(netlist)

    # Run simulation.
    ngs.exec_command("tran 1ns 100ns")
    ngs.run(background=False)

    # Retreive signals.
    plot_name = ngs.last_plot
    print(plot_name)
    if plot_name == 'const':
        raise Exception("Simulation failed.")
    plot = ngs.plot(None, plot_name)
    analysis = plot.to_analysis()

    plt.plot(analysis.time, analysis['y'], 'x-')
    plt.plot(analysis.time, np.array(analysis['a']), 'x-')
    plt.show()