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
from typing import Any, Dict, Set, List
from itertools import product


class MinAreaRouter(SignalRouter):
    """ Wrap a detail router function such that it respects minimum area constraints.

    Parameters
    ----------

    router: Underlying detail router
    node_conflict: Dict[Node, Set[Node]]
            A mapping from a node `n` to a set of nodes that collide with `n`.
    """

    def __init__(self, router: SignalRouter, node_conflict: Dict[Any, Set], min_area: Dict[Any, float]):
        self.router = router
        self.node_conflict = node_conflict
        self.min_area = min_area

    def route(self, G: nx.Graph,
              terminals: List[List[Any]],
              node_cost_fn,
              edge_cost_fn
              ) -> nx.Graph:
        return self._min_area_route(self.router,
                                    G,
                                    terminals,
                                    self.node_conflict,
                                    node_cost_fn,
                                    edge_cost_fn)

    def _min_area_route(self, router: SignalRouter, G: nx.Graph, terminals, node_conflict: Dict, node_cost_fn,
                        edge_cost_fn):
        S1 = router.route(G, terminals, node_cost_fn, edge_cost_fn)

        nodes = list(S1.nodes)

        # Get layers for which min_area is defined.
        min_area_layers = set(self.min_area.keys())

        return S1


