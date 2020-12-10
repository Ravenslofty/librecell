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
import networkx as nx
from itertools import combinations, permutations, chain, product

from typing import List, Set

import logging

logger = logging.getLogger(__name__)


def construct_even_degree_graphs(G: nx.MultiGraph) -> List[nx.MultiGraph]:
    """ Construct all graphs of even degree by inserting a minimal number of virtual edges.
    :param G: A nx.MultiGraph
    :return: List[nx.MultiGraph]
        Returns a list of all graphs that can be constructed by inserting virtual edges.
    """

    assert isinstance(G, nx.MultiGraph), Exception("G must be a nx.MultiGraph.")

    if nx.is_empty(G):
        logger.debug("Graph is empty.")
    else:
        # nx.is_connected is not defined for empty graphs.
        if not nx.is_connected(G):
            logger.debug("Graph is not connected. Assuming there is a transmission gate.")

    # Find nodes with odd degree.
    odd_degree_nodes = [n for n, deg in G.degree if deg % 2 == 1]

    assert len(odd_degree_nodes) % 2 == 0

    if len(odd_degree_nodes) == 0:
        # All node degrees are already even. Nothing to do.
        assert nx.is_connected(G), Exception("G must be a connected graph.")
        return [G.copy()]

    """
    Finding all even degree graphs by inserting a minimal number of edges works as follows:
    * Find nodes of odd degree `odd_degree_nodes` and find all possible pairings of them.
        * Find all dual partitionings of `odd_degree_nodes`
        * For each partitioning find all pairings across the two partitions.
    """

    # Find all dual partitionings of `odd_degree_nodes`
    partitions2 = list(combinations(odd_degree_nodes, len(odd_degree_nodes) // 2))
    partitions_a = partitions2[:len(partitions2) // 2]
    partitions_b = partitions2[:len(partitions2) // 2 - 1:-1]

    # Assert that a and b are complementary
    for a, b in zip(partitions_a, partitions_b):
        assert set(chain(a, b)) == set(odd_degree_nodes)
        assert set(a) & set(b) == set(), "Partitions must be disjoint!"

    even_degree_graphs = []
    # For each partitioning ...
    for partition_a, partition_b in zip(partitions_a, partitions_b):
        # ... find all pairings across the two partitions.
        for partition_b_permutation in permutations(partition_b):
            G2 = G.copy()
            for a, b in zip(partition_a, partition_b_permutation):
                assert G2.degree(a) % 2 == 1
                assert G2.degree(b) % 2 == 1
                G2.add_edge(a, b)

            for n, deg in G2.degree:
                assert deg % 2 == 0

            if nx.is_connected(G2):
                # In presence of transmission gates the resulting graph might not be connected.
                # Only store the graphs that are connected.
                # TODO: This might not scale if there are many transmission gates => Handle transmission gates more efficiently.
                even_degree_graphs.append(G2)

    for g1, g2 in combinations(even_degree_graphs, 2):
        assert g1 != g2, "There should be no duplicates."

    return even_degree_graphs


def find_all_euler_tours(G: nx.MultiGraph, start_node=None, end_node=None, visited_edges: Set = None):
    """ Find some tour starting at `start_node`.
    If `end_node` is given the trace will end there. However, it will not be a full tour.

    Parameters
    ----------
    G: The graph

    start_node: Start of the tour.

    visited_edges: Edges that should not appear in the tour anymore.

    Returns
    -------
    Tour through G starting and ending at `start_node`.
    """
    for n, deg in G.degree():
        assert deg % 2 == 0, Exception("All nodes in G must have even degree.")

    tours = []

    if visited_edges is None:
        visited_edges = set()

    if start_node is None:
        # Deterministically choose some start node.
        start_node = min(G.nodes)

    if end_node is None:
        end_node = start_node

    edges = list(G.edges(start_node, keys=True))

    assert len(edges) > 0

    for e in edges:
        a, b, c = e
        # Normalize edge by sorting start and end.
        ao, bo = tuple(sorted((a, b)))
        e_norm = ao, bo, c
        if e_norm not in visited_edges:
            if len(visited_edges) == len(G.edges) - 1:
                # Last edge
                assert b == end_node
                tours.append([(a, b, c)])
            else:
                assert len(visited_edges) < len(G.edges) - 1
                visited_edges_sub = visited_edges | {e_norm}
                start_sub = b
                sub_tours = find_all_euler_tours(G, start_sub, end_node, visited_edges=visited_edges_sub)

                tours.extend([[(a, b, c)] + s for s in sub_tours])

    return tours


def multigraph_networkx2rust(G):
    """ Convert a networkx MultiGraph into a edge list that can be processed by pyo3-cell.
    """

    # Map nodes to indices
    nodemap = {n: i for i, n in enumerate(G.nodes)}

    rustgraph = nx.MultiGraph()
    rustgraph.add_nodes_from((nodemap[n] for n in G.nodes))
    rustgraph.add_edges_from(((nodemap[a], nodemap[b], d) for a, b, d in G.edges))

    rust_edges = list(rustgraph.edges)

    return rust_edges, nodemap


def multigraph_rust2networkx(rust_routing_trees, nodemap):
    """ Inverse transformation of `multigraph_networkx2rust`.
    """

    reverse_map = {v: k for k, v in nodemap.items()}

    # Convert back to python nodes.
    routing_edges = [
        [(reverse_map[a], reverse_map[b]) for a, b in rt]
        for rt in rust_routing_trees
    ]
    routing_trees = []
    for re in routing_edges:
        rt = nx.Graph()
        rt.add_edges_from(re)
        routing_trees.append(rt)

    return routing_trees


if __name__ == '__main__':
    import pyo3_cell

    G = nx.MultiGraph()
    G.add_edge('a', 'b')
    # G.add_edge('b','c')
    # G.add_edge('b','d')
    # G.add_edge('b','e')
    # G.add_edge('c','f')
    # G.add_edge('d','f')
    # G.add_edge('e','f')
    G.add_edge('f', 'a')
    G.add_edge('g', 'a')

    G = construct_even_degree_graphs(G)[0]

    # euler_tour = find_euler_tour(G)

    # print(euler_tour)

    all_tours = find_all_euler_tours(G, 'a', 'a')

    rust_graph, node_map = multigraph_networkx2rust(G)
    print(rust_graph)
    all_tours_pyo3 = pyo3_cell.find_all_euler_tours(rust_graph, node_map['a'])

    print(len(all_tours))
    print(len(all_tours_pyo3))
    print(all_tours)
    print()
    print(all_tours_pyo3)
