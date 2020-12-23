import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os


def test_simple_simulation():
    """Run a simple transient simulation in a ngspice subprocess and retreive the results from a file."""

    sim_file = tempfile.mktemp()
    sim_output_file = tempfile.mktemp()

    spice_simulation_netlist = f"""
*
.title Simple RC circuit.

R1 VDD Y 1k
C1 Y GND 1u ic=0V
Vsrc_vdd VDD GND PWL(0 0 1ms 0V 2ms 1V)

.control
*option abstol=10e-15
*option reltol=10e-11
set filetype=ascii
* Enable output of vector names in the first line.
set wr_vecnames
tran 1ms 10ms
wrdata {sim_output_file} v(VDD) v(Y)
exit
.endc

.end
"""
    print(f"Write simulation file: {sim_file}")
    open(sim_file, 'w').write(spice_simulation_netlist)

    # Run simulation.
    ret = subprocess.run(["ngspice", sim_file])
    print(f"Subprocess return value: {ret}")
    if ret.returncode != 0:
        print(f"ngspice simulation failed: {ret}")
    assert ret.returncode == 0

    print(f"Read output data: {sim_output_file}")
    data = np.loadtxt(sim_output_file, skiprows=1)  # Skip the header.

    a_time = data[:, 0]
    a = data[:, 1]

    y_time = data[:, 2]
    y = data[:, 3]

    assert all(a_time == y_time)

    plt.plot(a_time, a, 'x-')
    plt.plot(y_time, y, 'x-')
    plt.show()

    os.remove(sim_output_file)
    os.remove(sim_file)
