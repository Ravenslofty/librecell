# # OBSOLETE: Interesting approach, but not up to date with the rest of the program.
# #
# # Copyright (c) 2019-2020 Thomas Kramer.
# #
# # This file is part of librecell
# # (see https://codeberg.org/tok/librecell).
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU Affero General Public License as
# # published by the Free Software Foundation, either version 3 of the
# # License, or (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU Affero General Public License for more details.
# #
# # You should have received a copy of the GNU Affero General Public License
# # along with this program. If not, see <http://www.gnu.org/licenses/>.
# #
# from scipy import optimize
#
# from PySpice.Spice.Netlist import Circuit, SubCircuit
#
# from PySpice.Spice.BasicElement import Mosfet
#
# import PySpice.Logging.Logging as Logging
#
# from ..liberty.util import get_pin_information
# from ..characterization.util import read_trip_points_from_liberty
# from lccommon.net_util import load_subcircuit
# from ..characterization.timing_combinatorial import measure_comb_cell
# from liberty.parser import parse_liberty
# from liberty.types import select_cell
# import argparse
# from typing import List
#
# import numpy as np
# import os
# import logging
#
# pyspice_logger = Logging.setup_logging()
#
# logger = logging.getLogger(__name__)
#
#
# def main():
#     """
#     Command-line tool for automated transistor sizing.
#     Currently only combinatorial cells are supported excluding tri-state cells.
#     :return:
#     """
#
#     parser = argparse.ArgumentParser(
#         description='Optimize transistor sizes of a combinatorial cell.'
#                     'Currently more a proof-of-concept than a useful tool.',
#         epilog='')
#
#     parser.add_argument('-l', '--liberty', required=True, metavar='LIBERTY', type=str,
#                         help='Liberty file. Used for definition of trip points.')
#
#     parser.add_argument('--cell', required=True, metavar='CELL_NAME', type=str, help='Cell name.')
#
#     parser.add_argument('--spice', required=True, metavar='SPICE', type=str,
#                         help='SPICE netlist containing a subcircuit with the same name as the cell.')
#
#     parser.add_argument('-I', '--include', required=False, action='append', metavar='SPICE_INCLUDE', type=str,
#                         help='SPICE files to include such as transistor models.')
#
#     parser.add_argument('-o', '--output', required=True, metavar='SPICE_OUT', type=str,
#                         help='Output netlist with sized transistors.')
#
#     parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
#
#     # Parse arguments
#     args = parser.parse_args()
#
#     DEBUG = args.debug
#     log_level = logging.DEBUG if DEBUG else logging.INFO
#
#     logging.basicConfig(format='%(module)16s %(levelname)8s: %(message)s', level=log_level)
#
#     cell_name = args.cell
#     lib_file = args.liberty
#     spice_file = args.spice
#     spice_output_file = args.output
#
#     # TODO: find ground and supply net names from spice netlist or liberty.
#     ground = 'gnd'
#     supply = 'vdd'
#
#     # Load subcircuit of cell.
#     logger.info("Load subcircuit: {}".format(spice_file))
#     subcircuit_raw = load_subcircuit(spice_file, cell_name)
#     subcircuit = subcircuit_raw.build()
#
#     logger.info("Reading liberty: {}".format(lib_file))
#     with open(lib_file) as f:
#         data = f.read()
#     library = parse_liberty(data)
#
#     cell_group = select_cell(library, cell_name)
#     assert cell_group.args[0] == cell_name
#     logger.info("Cell: {}".format(cell_name))
#
#     # Get information on pins
#     input_pins, output_pins, output_functions = get_pin_information(cell_group)
#
#     # TODO: let user choose parameters.
#     # Fixed channel length
#     channel_length = 5e-08
#     # Minimal and maximal channel width.
#     min_transistor_width = 0.1e-6
#     max_transistor_width = 1e-6
#
#     # Load operation voltage and temperature.
#     # TODO: load voltage/temperature from operating_conditions group
#     supply_voltage = library['nom_voltage']
#     temperature = library['nom_temperature']
#     logger.info('Supply voltage = {:f} V'.format(supply_voltage))
#     logger.info('Temperature = {:f} V'.format(temperature))
#
#     # Read trip points from liberty file.
#     trip_points = read_trip_points_from_liberty(library)
#
#     circuit = Circuit('Test circuit', ground=ground)
#
#     spice_include_files = args.include
#
#     # Load include files.
#     for inc in spice_include_files:
#         logger.info("Include '{}'".format(inc))
#
#         if os.path.isfile(inc):
#             circuit.include(inc)
#         else:
#             msg = "No such file: {}".format(inc)
#             logger.error(msg)
#             assert False, msg
#
#     # Power supply.
#     circuit.V('power_vdd', supply, circuit.gnd, supply_voltage @ u_V)
#
#     # Instantiate mosfets
#     model_map = {
#         'nmos': 'NMOS_VTL',
#         'pmos': 'PMOS_VTL'
#     }
#
#     def create_subcircuit(mosfet_sizes: List[float]) -> SubCircuit:
#         """
#         Create a clone of the original subcircuit with adapted transistor sizes.
#         :param mosfet_sizes:
#         :return:
#         """
#         sc = SubCircuit(subcircuit.name, *subcircuit.nodes)
#         mosfets = (e for e in subcircuit.elements if isinstance(e, Mosfet))
#         for i, (t, size) in enumerate(zip(mosfets, mosfet_sizes)):
#             right, gate, left, body = t.nodes
#
#             model = model_map[t.model]
#
#             # TODO: floating body for SOI
#             sc.M(i + 1, right, gate, left, body,
#                  model=model,
#                  width=size,
#                  length=channel_length
#                  )
#
#         return sc
#
#     # mosfets = []
#     # for i, t in enumerate((e for e in subcircuit.elements if isinstance(e, Mosfet))):
#     #     right, gate, left, body = t.nodes
#     #
#     #     model = model_map[t.model]
#     #
#     #     # TODO: floating body for SOI
#     #     m = circuit.M(i + 1, right, gate, left, body,
#     #                   model=model,
#     #                   width=t.width,
#     #                   length=channel_length
#     #                   )
#     #     mosfets.append(m)
#
#     time_resolution = 10 @ u_ps
#
#     def f(widths: List[float]) -> float:
#         """
#         Objective function to me minimized.
#
#         Creates a version of the circuit with the given transistor sizing, simulates it and
#         returns a cost metric.
#         :param widths: Transistor widths. Ordering of widths must correspond to the ordering
#             of MOSFETs in the original subcircuit.
#         :return: Cost of the transistor sizing.
#         """
#
#         # TODO: let user define load capacitance and input slew.
#         input_transition_time = 0.06
#         output_cap = 1.0
#
#         _circuit = circuit.clone()
#         cut = create_subcircuit(widths)
#         _circuit.subcircuit(cut)
#         _circuit.X('CUT', cut.name, *subcircuit.nodes)
#
#         # for mosfet, width in zip(mosfets, widths):
#         #     mosfet.width = width
#
#         print(_circuit)
#
#         objective = 0
#         for output in output_pins:
#             for active_input in input_pins:
#                 r = measure_comb_cell(_circuit,
#                                       inputs_nets=input_pins,
#                                       active_pin=active_input,
#                                       output_net=output,
#                                       output_functions=output_functions,
#                                       input_rise_time=input_transition_time @ u_ns,
#                                       input_fall_time=input_transition_time @ u_ns,
#                                       output_load_capacitance=output_cap @ u_pF,
#                                       vdd=supply_voltage,
#                                       temperature=temperature,
#                                       trip_points=trip_points,
#                                       time_step=time_resolution,
#                                       simulation_duration_hint=1000 @ u_ps)
#
#             diff = ((r['cell_fall'] - r['cell_rise']) * 1e9) ** 2
#             # _objective = diff + r['cell_rise'] * 1e6 + r['cell_fall'] * 1e6
#             #sum2 = (r['cell_fall'] * 1e9) ** 2 + (r['cell_rise'] * 1e9) ** 2
#
#             _objective = diff  # + sum2
#             # objective = diff[0] + r['rise_delay'] * 1e9
#
#             objective += _objective
#
#         # print(objective)
#         return objective
#
#     start_width = np.mean([min_transistor_width, max_transistor_width])
#
#     num_mosfets = len([e for e in subcircuit.elements if isinstance(e, Mosfet)])
#
#     optimal_widths = optimize.minimize(f,
#                                        x0=[start_width] * num_mosfets,
#                                        bounds=[(min_transistor_width, max_transistor_width)] * num_mosfets,
#                                        tol=3e-7)
#
#     if optimal_widths.success:
#
#         logger.info("Optimization done: {}".format(optimal_widths.message))
#         logger.info("Total function evaluations: {}".format(optimal_widths.nfev))
#         final_subcircuit = create_subcircuit(optimal_widths.x)
#
#         with open(spice_output_file, 'w') as f:
#             logger.info("Write SPICE netlist: {}".format(spice_output_file))
#             f.write(str(final_subcircuit))
#     else:
#         logger.error("Optimization failed: {}".format(optimal_widths.message))
#         exit(1)
