from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Parser import SpiceParser
from PySpice.Unit import *
from PySpice.Unit.SiUnits import Farad, Second, Volt
from typing import Any, Dict, Union, Optional

from .piece_wise_linear import PieceWiseLinear


def simulate_circuit(circuit: Circuit,
                     input_voltages: Dict[Any, Union[float, PieceWiseLinear]],
                     time_step: Second,
                     end_time: Second,
                     temperature: float = 25,
                     initial_voltages: Optional[Dict[str, Volt]] = None):
    """
    Simulate a circuit with given input voltages.
    :param circuit:
    :param input_voltages: Dict[node name, input voltage]
        Input voltages can either be a `float` for constant values or a `.util.PieceWiseLinearWave`
        changing wave forms.
    :param time_step: Time step of simulation.
    :param end_time: End time of simulation.
    :param temperature: Simulation temperature.
    :param initial_voltages: Dict[node name, voltage].
        Initial voltages for nodes. Default = None.
    :return: PySpice Analysis object.
    """
    circuit = circuit.clone()

    ngspice_shared = NgSpiceShared.new_instance(send_data=False)
    simulator = circuit.simulator(temperature=temperature,
                                  nominal_temperature=temperature,
                                  simulator='ngspice-shared',
                                  ngspice_shared=ngspice_shared
                                  )

    # Create input drivers.
    for name, voltage in input_voltages.items():
        if isinstance(voltage, PieceWiseLinear):
            piece_wise_linear_voltage_source(circuit, 'in_{}'.format(name), name, circuit.gnd,
                                             voltage
                                             )
            # simulator.initial_condition(**{name: voltage(0) @ u_V})
        else:
            # simulator.initial_condition(**{name: voltage @ u_V})
            circuit.V('in_{}'.format(name), name, circuit.gnd, voltage @ u_V)

    if initial_voltages is not None:
        simulator.initial_condition(**initial_voltages)

    # Run transient analysis.
    # Set use_initial_condition (uic) to False to enable DC bias computation. See ngspice manual 15.2.2 2)
    analysis = simulator.transient(step_time=time_step,
                                   end_time=end_time,
                                   use_initial_condition=False
                                   )
    return analysis


def piece_wise_linear_voltage_source(circuit: Circuit, name: str, plus, minus, wave: PieceWiseLinear,
                                     repeat=None, time_delay=None):
    """ Create a piece wise linear voltage source.
    This is a helper function needed because PWL sources are not properly handled by PySpice at time of this writing.
    :param circuit: The netlist to add the source.
    :param name: Name of the voltage source.
    :param plus: Positive terminal.
    :param minus: Negative terminal.
    :param times: Time axis.
    :param voltages: Voltage axis.
    :param repeat: Number of times the sequence should be repeated. Default = 1. 0 means endless repetition.
    :param time_delay: Delay the signal by this amount of time. Default = 0.
    :return: Return a handle to the generated voltages source.
    """
    pwl_string = ' '.join((
        '%es %eV' % (float(time), float(voltage))
        for time, voltage in zip(wave.x, wave.y)
    ))

    pwl_args = []
    if repeat is not None:
        pwl_args.append('r={}'.format(repeat))
    if time_delay is not None:
        pwl_args.append('td={}'.format(float(time_delay)))

    return circuit.V(name, plus, minus,
                     'PWL({}) DC=0'.format(' '.join([pwl_string] + pwl_args))
                     )
