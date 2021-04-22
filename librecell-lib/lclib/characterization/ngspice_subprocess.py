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
Simple sub-process based NgSpice binding.
"""

import os
import subprocess
import logging
import numpy as np
from typing import Dict, List, Tuple, Union
from .piece_wise_linear import PieceWiseLinear

logger = logging.getLogger(__name__)


def run_simulation(sim_file: str, ngspice_executable: str = 'ngspice'):
    """
    Invoke 'ngspice' to run the `sim_file`.
    :param sim_file: Path to ngspice simulation file.
    :return: Returns (stdout, stderr) outputs of ngspice.
    """
    logger.debug(f"Run simulation: {sim_file}")
    try:
        ret = subprocess.run([ngspice_executable, sim_file], capture_output=True)
        # proc = subprocess.Popen([ngspice_executable, sim_file])
        # logger.debug(f"Subprocess return value: {ret}")
        if ret.returncode != 0:
            ngspice_err_message = ret.stderr.decode("utf-8")
            logger.error(f"ngspice simulation failed: {ngspice_err_message}")
            raise Exception(f"ngspice simulation failed: {ngspice_err_message}")

        return ret.stdout.decode("utf-8"), ret.stderr.decode("utf-8")
    except FileNotFoundError as e:
        msg = f"SPICE simulator executable not found. Make sure it is in the current path: {ngspice_executable}"
        logger.error(msg)
        raise FileNotFoundError(msg)


def simulate_cell(
        cell_name: str,
        cell_ports: List[str],
        input_voltages: Dict[str, Union[PieceWiseLinear, float]],
        initial_voltages: Dict[str, float],
        breakpoint_statements: List[str],
        output_voltages: List[str],
        output_currents: List[str],
        simulation_file: str,
        simulation_output_file: str,
        max_simulation_time: float,
        simulation_title: str = "<UNTITLED SIMULATION>",
        temperature: float = 25,
        output_load_capacitances: Dict[str, float] = None,
        time_step: float = 100.0e-12,
        setup_statements: List[str] = None,
        ground_net: str = 'GND',
        debug: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Simulate a circuit with given input signals and measure voltages and currents.

    :param cell_name: Name of the cell to be characterized. Must match with the name used in netlist and liberty.
    :param cell_ports: All circuit pins/ports in the same ordering as used in the SPICE circuit model.
    :param input_voltages: Dictionary holding input voltages or waveforms per net.
    :param initial_voltages: Additional initial voltages per net name at the beginning of the simulation.
        By default initial voltages are derived from the input voltages. This is useful for specifying initial voltages of outputs.
    :param breakpoint_statements: List of ngspice breakpoint statements.
    :param output_voltages: List of voltages that should be in the output of this function.
    :param output_currents: List of sources whose current should be in the output of this function.
    :param simulation_file: Path to the SPICE file which will be generated.
    :param simulation_output_file: Path of the simulation output data.
    :param max_simulation_time: Maximal simulation time in seconds.
    :param simulation_title: Title of the simulation that will be written into the simulation file.
    :param temperature: Temperature of the simulation.
    :param output_load_capacitances: A dict with (net, capacitance) pairs which defines the load capacitances attached to certain nets.
    :param time_step: Simulation time step.
    :param setup_statements: SPICE statements that are included at the beginning of the simulation.
        This should be used for .INCLUDE and .LIB statements.
    :param ground_net: The name of the ground net.
    :param debug: Enable more verbose debugging output such as plots of the simulations.
    :return: Returns tuple (time, Dict[net, voltage], Dict[source, current]).
    The output voltages and currents correspond to the values specified in `output_voltages` and `output_currents`.
    """

    logger.debug(f"Simulate cell: {cell_name}")
    logger.debug(f"Cell ports: {cell_ports}")
    logger.debug(f"Initial voltages: {initial_voltages} [V]")
    logger.debug(f"Ground net: {ground_net}")
    logger.debug(f"Temperature: {temperature} C")
    logger.debug(f"Max. simulation time: {max_simulation_time} s")

    # Create a list of include files.
    if setup_statements is None:
        setup_statements = []

    # Load include files.
    setup_statements_string = "\n".join(setup_statements)

    input_voltages_static = dict()
    input_voltages_active = dict()

    # Split input signals into static and active signals
    for net, voltage in input_voltages.items():
        if isinstance(voltage, PieceWiseLinear):
            input_voltages_active[net] = voltage
        else:
            assert isinstance(voltage, float), "Voltage needs to be a float or a piecewise linear function."
            input_voltages_static[net] = voltage

    logger.debug("Static input voltages: {}".format(sorted(list(input_voltages_static.keys()))))
    logger.debug("Dynamic input voltages: {}".format(sorted(list(input_voltages_active.keys()))))

    def create_voltage_source_statement(net: str, voltage: Union[float, PieceWiseLinear]) -> str:
        """
        Create a SPICE statement for a voltage source driving the 'net' with a voltage.
        The voltage can be either a static value or a `PieceWiseLinear` function.
        """
        if isinstance(voltage, PieceWiseLinear):
            return f"V{net} {net} {ground_net} PWL({voltage.to_spice_pwl_string()}) DC=0"
        elif isinstance(voltage, float):
            return f"V{net} {net} {ground_net} {voltage}"
        else:
            assert False, "`voltage` must be either a float or {}".format(PieceWiseLinear)

    # Create SPICE description of the input voltage sources.
    active_signal_source_statements = "\n".join(
        (
            create_voltage_source_statement(net, wave)
            for net, wave in input_voltages_active.items()
        )
    )

    # Static inputs.
    static_signal_source_statements = "\n".join(
        (
            create_voltage_source_statement(net, voltage)
            for net, voltage in input_voltages_static.items()
        )
    )

    # Load capacitance statements.
    if output_load_capacitances is None:
        output_load_capacitances = dict()
    else:
        assert isinstance(output_load_capacitances, dict)
    load_capacitance_statements = "\n".join(
        (
            f"Cload_{net} {net} {ground_net} {load}"
            for net, load in output_load_capacitances.items()
        )
    )

    # Initial node voltages.
    initial_conditions = initial_voltages

    # Calculate initial conditions for all sources.
    for net, voltage in input_voltages.items():
        if isinstance(voltage, PieceWiseLinear):
            initial_conditions[net] = voltage(0)
        else:
            assert isinstance(voltage, float)
            initial_conditions[net] = voltage

    # Format breakpoint statements.
    breakpoint_statements = '\n'.join(breakpoint_statements)

    # Generate output definitions for voltages.
    output_voltage_defs = " ".join((f"v({v})" for v in output_voltages))

    # Can measure currents only through sources.
    # Check that specified output currents match with specified voltage sources.
    for current in output_currents:
        if current not in input_voltages:
            msg = "Can measure currents only through voltage sources. " \
                  f"'{current}' does not correspond to a source."
            logger.error(msg)
            assert False, msg

    # Generate output definitions for currents.
    output_current_defs = " ".join((f"i(v{source})" for source in output_currents))

    # Create ngspice simulation script.
    sim_netlist = f"""* librecell {__name__}
.title {simulation_title}

.option TEMP={temperature}

{setup_statements_string}

Xcircuit_under_test {" ".join(cell_ports)} {cell_name}

* Output load capacitances.
{load_capacitance_statements}

* Static input and supply voltages.
{static_signal_source_statements}

* Active input signals (clock & data_in).
{active_signal_source_statements}

* Initial conditions.
* Also all voltages of DC sources must be here if they are needed to compute the initial conditions.
.ic {" ".join((f"v({net})={v}" for net, v in initial_conditions.items()))}

.control 
*option reltol=1e-5
*option abstol=1e-15

set filetype=ascii
set wr_vecnames

* Breakpoints
{breakpoint_statements}

* Transient simulation, use initial conditions.
tran {time_step} {max_simulation_time} uic
* Write selected signals to the output file.
wrdata {simulation_output_file} {output_voltage_defs} {output_current_defs} 
* Exit ngspice.
exit
.endc

.end
"""

    # Dump simulation script to the file.
    logger.debug(f"Write simulation netlist: {simulation_file}")
    if os.path.exists(simulation_file):
        logger.warning("Simulation file already exists: {}".format(simulation_file))
    open(simulation_file, "w").write(sim_netlist)

    # Start ngspice.
    logger.debug("Run simulation.")
    stdout, stderr = run_simulation(simulation_file)

    # Retrieve data.
    logger.debug("Load simulation output.")
    sim_data = np.loadtxt(simulation_output_file, skiprows=1)

    if sim_data.ndim != 2:
        logger.error("Simulation failed. No data was written to the output file.")
        if debug:
            logger.error(f"ngspice: {stderr}")
        assert False, "Simulation failed. No data was written to the output file."

    # Extract data from the numpy array.
    time = sim_data[:, 0]
    index = 1

    # Put voltages into a dict.
    voltages = dict()
    for v_out in output_voltages:
        voltages[v_out] = sim_data[:, index]
        index = index + 2

    # Put currents into a dict.
    currents = dict()
    for i_out in output_currents:
        currents[i_out] = sim_data[:, index]
        index = index + 2

    return time, voltages, currents
