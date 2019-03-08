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
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Parser import SpiceParser
from PySpice.Unit import *
from PySpice.Unit.SiUnits import Farad, Second

import PySpice.Logging.Logging as Logging
import numpy as np

from .ngspice_simulation import piece_wise_linear_voltage_source
from .util import PieceWiseLinear


def test_simulate_circuit():
    time_step = 1 @ u_ms
    end_time = 10 @ u_s

    circuit = Circuit(title='test')

    circuit.V('VDD', 'vdd', circuit.gnd, 10 @ u_V)
    circuit.R('1', 'input', 'a', 1 @ u_kOhm)
    circuit.C('1', circuit.gnd, 'a', 1 @ u_nF)

    ngspice_shared = NgSpiceShared.new_instance(send_data=False)

    piece_wise_linear_voltage_source(circuit, 'src', circuit.gnd, 'input',
                                     PieceWiseLinear([0, 1, 2, 10], [0, 1, 1, 1]))

    simulator = circuit.simulator(temperature=25,
                                  nominal_temperature=25,
                                  simulator='ngspice-shared',
                                  ngspice_shared=ngspice_shared)

    ngspice_shared.stop('v(a) > 10')
    # ngspice_shared.destroy()
    # print(str(simulator))
    # ngspice_shared.load_circuit(str(simulator))
    # ngspice_shared.run()
    # analysis = ngspice_shared.plot(simulator, 'asdf').to_analysis()

    simulator = circuit.simulator(temperature=25,
                                  nominal_temperature=25,
                                  simulator='ngspice-shared',
                                  ngspice_shared=ngspice_shared)
    analysis = simulator.transient(step_time=time_step, end_time=end_time)

    print(np.array(analysis['a']))
