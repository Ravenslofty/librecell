
import logging
logger = logging.getLogger(__name__)

from .data_types import *

from itertools import chain


import pyo3_cell as pc
	
	
def transistors2pyo3(transistors):
	
	ids = {t: i for i,t in enumerate(transistors) }
	
	# Convert nets to integer IDs.
	all_nets = set(chain(*((t.left, t.gate, t.right) for t in transistors)))
	net_map = {n: i for i,n in enumerate(all_nets)}
	
	l = [(net_map[t.left], net_map[t.gate], net_map[t.right], t.channel_type == ChannelType.PMOS, ids[t]) 
		for t in transistors]

	reverse_net_map = {i: n for n,i in net_map.items()}
	reverse_transistor_map = {i: t for t,i in ids.items()}
	
	return l, reverse_net_map, reverse_transistor_map
		
	
	
if __name__ == '__main__':
	from lccommon import net_util
	
	cell_name = "OR2X1"
	path = "/home/user/FreePDK45/osu_soc/lib/source/netlists/" + cell_name + ".pex.netlist"

	transistors, io_pins = net_util.load_transistor_netlist(path, cell_name)
	
	pyo3_netlist, reverse_net_map, reverse_transistor_map = transistors2pyo3(transistors)
	
	print(pyo3_netlist)
	
	rows = pc.place_cell(pyo3_netlist)
	
	for row in rows:
		print(row)
	
