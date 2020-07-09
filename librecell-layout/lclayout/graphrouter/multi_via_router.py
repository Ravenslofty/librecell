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
from .signal_router import SignalRouter
import networkx as nx
from typing import Any, Dict, Set
from itertools import product


class MultiViaRouter(SignalRouter):

    def __init__(self, router: SignalRouter, node_conflict: Dict[Any, Set]):
        """
        Wrap a detail router function such that it can create double vias.

        Parameters
        ----------

        :param router: Underlying detail router
        :param node_conflict: Dict[Node, Set[Node]]
                A mapping from a node `n` to a set of nodes that collide with `n`.
        """
        self.router = router
        self.node_conflict = node_conflict if node_conflict is not None else dict()

    def route(self, graph: nx.Graph,
              terminals,
              node_cost_fn,
              edge_cost_fn
              ) -> nx.Graph:
        return _multi_via_spanning_subtree(self.router,
                                                graph,
                                                terminals,
                                                self.node_conflict,
                                                node_cost_fn,
                                                edge_cost_fn)


def _multi_via_spanning_subtree(router: SignalRouter,
                                graph: nx.Graph,
                                terminals,
                                node_conflict: Dict, node_cost_fn,
                                edge_cost_fn):
    """ Wrap a detail router function such that it can create double vias.

    Parameters
    ----------

    router: Underlying detail router
    graph: nx.Graph
    terminals: The nodes to connect.
    node_collisions: Dict[Node, Set[Node]]
            A mapping from a node `n` to a set of nodes that collide with `n`.
    """

    S1 = router.route(graph, terminals, node_cost_fn, edge_cost_fn)

    multi_vias = set()
    for a, b in S1.edges:
        multi_via = graph[a][b].get('multi_via', 1)
        if multi_via > 1:
            multi_vias.add(((a, b), multi_via))

    # Remove edges from Graph
    if multi_vias:
        G2 = graph.copy()

        for a, b in S1.edges():
            G2.add_edge(a, b, weight=0.0001)

        for (a, b), n in multi_vias:

            a_equiv = node_conflict.get(a, [])
            b_equiv = node_conflict.get(b, [])

            for c, d in product(a_equiv, b_equiv):
                if (c, d) in G2.edges:
                    G2.remove_edge(c, d)

        # Reconnect the disconnected nodes.
        for (a, b), n in multi_vias:
            # Connect two nodes.
            path = router.route(G2, [a, b], node_cost_fn, edge_cost_fn)
            for e in path.edges():
                S1.add_edge(*e)

    return S1
