#
# Copyright 2019-2020 Thomas Kramer.
#
# This source describes Open Hardware and is licensed under the CERN-OHL-S v2.
#
# You may redistribute and modify this documentation and make products using it
# under the terms of the CERN-OHL-S v2 (https:/cern.ch/cern-ohl).
# This documentation is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY,
# INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
# Please see the CERN-OHL-S v2 for applicable conditions.
#
# Source location: https://codeberg.org/tok/librecell
#

import pyo3_cell
import networkx as nx

def route(G, signals, reserved_nodes = None, node_conflict=None):
	
	if isinstance(signals, dict):
		signals = list(signals.values())
	
	if isinstance(reserved_nodes, dict):
		reserved_nodes = list(reserved_nodes.values())
		
	if reserved_nodes is None:
		reserved_nodes = []*len(signals)
	
	rust_edges, nodemap = graph_networkx2rust(G)
	rust_signals = [[nodemap[t] for t in terminals] for terminals in signals]
	
	node_collisions = list(node_conflict.items()) if node_conflict else []
	node_collisions = [(nodemap[n], [nodemap[c] for c in collisions]) for n, collisions in node_collisions]
	
	reserved_nodes = [[nodemap[n] for n in nodes] for nodes in reserved_nodes]
	
	rust_routing_trees = pyo3_cell.route_mst(rust_edges, rust_signals, reserved_nodes, node_collisions)
	
	routing_trees = graph_rust2networkx(rust_routing_trees, nodemap)
	return routing_trees

def graph_networkx2rust(G):
	# Map nodes to indices
	nodemap = {n: i for i,n in enumerate(G.nodes)}
	
	rustgraph = nx.Graph()
	rustgraph.add_nodes_from((nodemap[n] for n in G.nodes))
	rustgraph.add_edges_from(((nodemap[a], nodemap[b], d) for a,b,d in G.edges(data=True)))
	
	rust_edges = list(rustgraph.edges(data=True))
	
	for a,b,data in rust_edges:
		assert data['weight'] > 0
	
	return rust_edges, nodemap
	
def graph_rust2networkx(rust_routing_trees, nodemap):

	reverse_map = {v:k for k,v in nodemap.items()}
	
	# Convert back to python nodes.
	routing_edges = [
		[(reverse_map[a], reverse_map[b]) for a,b in rt]
		for rt in rust_routing_trees
	]
	routing_trees = []
	for re in routing_edges:
		rt = nx.Graph()
		rt.add_edges_from(re)
		routing_trees.append(rt)
		
	return routing_trees
