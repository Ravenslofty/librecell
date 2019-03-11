##
## Copyright (c) 2019 Thomas Kramer.
## 
## This file is part of librecell-layout 
## (see https://codeberg.org/tok/librecell/src/branch/master/librecell-layout).
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the CERN Open Hardware License (CERN OHL-S) as it will be published
## by the CERN, either version 2.0 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## CERN Open Hardware License for more details.
## 
## You should have received a copy of the CERN Open Hardware License
## along with this program. If not, see <http://ohwr.org/licenses/>.
## 
## 
##
import networkx as nx
from itertools import chain, combinations, product

from typing import Any, Dict, Set, Tuple, AbstractSet, Optional

import logging
from .graphrouter import GraphRouter

logger = logging.getLogger(__name__)


class HVGraphRouter(GraphRouter):

    def __init__(self, sub_graphrouter: GraphRouter, orientation_change_penalty: float = 1):
        self.sub_graphrouter = sub_graphrouter
        self.orientation_change_penalty = orientation_change_penalty

    def route(self,
              graph: nx.Graph,
              signals: Dict[Any, AbstractSet[Any]],
              reserved_nodes: Optional[Dict] = None,
              node_conflict: Optional[Dict[Any, AbstractSet[Any]]] = None
              # node_cost_fn,
              # edge_cost_fn
              ) -> Dict[Any, nx.Graph]:
        return _route_hv(self.sub_graphrouter,
                         graph,
                         signals=signals,
                         reserved_nodes=reserved_nodes,
                         node_conflict=node_conflict)


def _build_hv_routing_graph(graph: nx.Graph, orientation_change_penalty=1) -> Tuple[nx.Graph, Dict, Dict]:
    """ Apply a transformation to G to simplify corner avoidance routing.
    Essentially nodes with edges of different orientations are split into multiple virtual nodes,
    one per edge orientation.
    The virtual nodes are connected by virtual edges of weight `orientation_change_penalty`.

    Use `flatten_hv_graph` to transform a graph/subgraph back to its original form.

    Parameters
    ----------
    graph: Routing graph with edge orientation information.
            Edge orientation must be stored in the 'orientation' field of the networkx edge data.
    orientation_change_penalty: Cost for changes between different orientations.

    :return : Tuple[nx.Graph, Dict[node, Dict[orientation, node]], Dict[node, node]]
    """

    assert nx.is_connected(graph), Exception("Graph G is not connected.")

    H = nx.Graph()

    # G -> H
    node_mapping = dict()

    # Create new nodes.
    for n1 in graph:
        edges = graph.edges(n1, data=True)
        orientations = set(data.get('orientation', None) for a, b, data in edges)
        # orientations = {o for o in orientations if o is not None}
        orientations.add(None)
        has_different_orientations = len(orientations) > 1

        # Add edges between virtual nodes.
        if has_different_orientations:
            # Create a virtual node for all orientations.
            n2s = {o: (o, n1) for o in orientations}
            node_mapping[n1] = n2s
            assert (len(n2s.values()) >= 2)
            # Mutually connect virtual nodes.
            for a, b in combinations(n2s.values(), 2):
                w = orientation_change_penalty
                if a is None or b is None:
                    w = 0
                H.add_edge(a, b, weight=w)
        else:
            # No need to split node.
            n2 = (None, n1)
            n2s = {None: n2}
            node_mapping[n1] = n2s
            H.add_node(n2)

    # Create new edges.
    for a, b, data in graph.edges(data=True):

        orientation = data.get('orientation', None)
        all_a2 = node_mapping[a]
        all_b2 = node_mapping[b]

        if orientation in all_a2 and orientation in all_b2:
            a2 = all_a2[orientation]
            b2 = all_b2[orientation]
            H.add_edge(a2, b2, **data)
        else:
            for a2, b2 in product(all_a2.values(), all_b2.values()):
                H.add_edge(a2, b2, **data)

    for n in H.nodes:
        assert nx.degree(H, n) > 0, Exception("Unconnected node %s" % str(n))

    assert nx.is_connected(H), Exception("Graph is not connected.")

    # Create reverse mapping.
    reverse_mapping = dict()
    for n1, n2s in node_mapping.items():
        for n2 in n2s.values():
            reverse_mapping[n2] = n1

    return H, node_mapping, reverse_mapping


def _flatten_hv_graph(hv_graph: nx.Graph, reverse_mapping: Dict) -> nx.Graph:
    """ Inverse transformation of `build_hv_routing_graph`.
    """

    G = nx.Graph()

    for n in hv_graph:
        n2 = reverse_mapping[n]
        G.add_node(n2)

    for a, b, data in hv_graph.edges(data=True):
        a2 = reverse_mapping[a]
        b2 = reverse_mapping[b]
        if a2 != b2:
            G.add_edge(a2, b2, **data)

    return G


def _route_hv(router: GraphRouter,
              graph: nx.Graph,
              signals: Dict[Any, AbstractSet[Any]],
              orientation_change_penalty: float = 1,
              node_conflict: Dict[Any, Set[Any]] = None,
              reserved_nodes: Optional[Dict[Any, AbstractSet[Any]]] = None,
              **kw) -> Dict[Any, nx.Graph]:
    """ Global routing with corner avoidance.
    Corners (changes between horizontal/vertical tracks) are avoided by transforming the routing graph `G`
    such that corners are penalized.

    Parameters
    ----------
    :param graph: Routing graph with edge orientation information.
            Edge orientation must be stored in the 'orientation' field of the networkx edge data.
    :param signals: A dict mapping signal names to signal terminals.
    :param orientation_change_penalty: Cost for changes between different orientations.
    :param reserved_nodes: An optional dict which specifies nodes that are reserved for a specific net.
    Dict[net_name, set of nodes].
    :param kw: Parameters to be passed to underlying routing function.

    """

    assert isinstance(signals, dict)
    logger.debug('Start global routing with corner avoidance.')

    H, node_mapping, node_mapping_reverse = _build_hv_routing_graph(
        graph,
        orientation_change_penalty=orientation_change_penalty
    )
    reserved_nodes_h = None
    if reserved_nodes is not None:
        reserved_nodes_h = {net: list(chain(*(node_mapping[n].values() for n in nodes))) for net, nodes in
                            reserved_nodes.items()}

    if node_conflict is None:
        node_conflict = dict()

    # Some nodes in H will be mapped to the same node in G and therefore conflict with each other.
    node_conflict_h = dict()
    # for n_h, n_g in node_mapping_reverse.items():
    #     node_conflict_h[n_h] = node_mapping[n_g].values()

    for n_h in H:
        n_g = node_mapping_reverse[n_h]

        conflicts = set()
        conflicts.update(node_mapping[n_g].values())

        if n_g in node_conflict:
            conflicts_g = set(node_conflict[n_g])
            for n in conflicts_g:
                conflicts.update(node_mapping[n].values())

        node_conflict_h[n_h] = conflicts

    signals_h = {net: [node_mapping[t][None] for t in terminals] for net, terminals in signals.items()}

    assert nx.is_connected(H)
    routing_trees_h = router.route(H, signals_h, reserved_nodes=reserved_nodes_h,
                                   node_conflict=node_conflict_h, **kw)

    # logger.info("Use pyo3 router.")
    # from . import pyo3_graphrouter
    # routing_trees_h = pyo3_graphrouter.route(H, signals_h, node_conflict=node_conflict_h)

    # Translate routing trees from H back to G.
    routing_trees = {name: _flatten_hv_graph(rt, node_mapping_reverse)
                     for name, rt in routing_trees_h.items()}

    # Assert that reserved_nodes is respected.
    # A node that is reserved for a signal should not be used by another signal.
    if reserved_nodes:
        for net, nodes in reserved_nodes.items():
            for rt_net, rt in routing_trees.items():
                if net != rt_net:
                    for n in nodes:
                        assert n not in rt.nodes, \
                            "Node %s is reserved for net %s but has been used for net %s." % (
                                n, net, rt_net)

    return routing_trees
