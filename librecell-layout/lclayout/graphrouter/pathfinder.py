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
from .graphrouter import GraphRouter
from .signal_router import SignalRouter
from .multi_via_router import MultiViaRouter

from itertools import chain, count, combinations, product
from collections import Counter

from typing import Any, Dict, List, AbstractSet, Optional

import logging

logger = logging.getLogger(__name__)


class PathFinderGraphRouter(GraphRouter):

    def __init__(self, detail_router: SignalRouter):
        self.detail_router = detail_router

    def route(self,
              graph: nx.Graph,
              signals: Dict[Any, List[Any]],
              reserved_nodes: Optional[Dict] = None,
              node_conflict: Optional[Dict[Any, AbstractSet[Any]]] = None
              # node_cost_fn,
              # edge_cost_fn
              ) -> Dict[Any, nx.Graph]:
        return _route(self.detail_router,
                      graph,
                      signals=signals,
                      reserved_nodes=reserved_nodes,
                      node_conflict=node_conflict)


def _route(detail_router: SignalRouter,
           graph: nx.Graph,
           signals: Dict[Any, List[Any]],
           reserved_nodes: Optional[Dict] = None,
           node_conflict: Optional[Dict[Any, AbstractSet[Any]]] = None) -> Dict[Any, nx.Graph]:
    """ Route multiple signals in the graph.
    Based on PathFinder algorithm.

    Parameters
    ----------
    :param detail_router: Underlying detail router.
    :param graph               :       networkx.Graph
                            Graph representing the routing grid.
    :param signals  :       Dict[node name, List[node]]
                            Signals to be routed. Each signal is represented by its terminal nodes.
    :param reserved_nodes: An optional dict which specifies nodes that are reserved for a specific net.
    Dict[net_name, set of nodes].
    :param node_conflict: Dict[node, Set[node]]
    Tells which other nodes are blocked by a node. A node might block its direct neigbhours to ensure minimum spacing.

    :returns : A list of `networkx.Graph`s representing the routes of each signal.
    """

    assert isinstance(signals, dict)

    logger.info('Start global routing.')

    assert nx.is_connected(graph), Exception("Cannot route in unconnected graph.")

    if reserved_nodes is None:
        reserved_nodes = dict()

    if node_conflict is None:
        node_conflict = dict()

    # Costs
    default_edge_cost = 1
    default_node_cost = 0.1
    edge_base_cost = {}  # Base cost of resource. Depends on layer type, geometries, ... (bn)

    node_base_cost = {}  # Base cost of resource. Depends on layer type, geometries, ... (bn)
    node_history_cost = {}  # Related to past congestion. (hn)
    node_present_sharing_cost = {}  # Related to number of signals using this resource in the current iteration. (pn)

    # Import edge costs from graph G
    for a, b, data in graph.edges(data=True):
        assert 'weight' in data, Exception('Edge has no weight: ', (a, b))
        if 'weight' in data:
            edge_base_cost[(a, b)] = data['weight']
            edge_base_cost[(b, a)] = data['weight']

    routing_trees = {name: nx.Graph() for name in signals.keys()}
    slack_ratios = {name: 1 for name in signals.keys()}

    # For each net create its own routing graph.
    # Some routing nodes are only available for a specific net.
    Gs = dict()
    all_reserved = set(chain(*reserved_nodes.values()))
    for net, terminals in signals.items():
        forbidden_nodes = all_reserved - set(reserved_nodes.get(net, {}))
        G2 = graph.copy()
        if forbidden_nodes:
            # Need to delete some nodes from G.
            G2.remove_nodes_from(forbidden_nodes)

            for t1, t2 in combinations(terminals, 2):
                assert nx.node_connectivity(G2, t1, t2) > 0, \
                    Exception("Graph has been disconnected by removal of reserved nodes.")

            Gs[net] = G2

    # TODO: just use detail_router and let caller decide whether to use MultiViaRouter or not.
    multi_via_router = MultiViaRouter(detail_router, node_conflict)

    max_iterations = 1000
    for j in count():

        if j >= max_iterations:
            raise Exception("Failed to route")

        logger.info('Routing iteration %d' % j)

        routing_order = sorted(signals.keys(), key=lambda i: slack_ratios[i], reverse=True)
        node_present_sharing_cost.clear()

        for signal_name in routing_order:
            terminals = signals[signal_name]

            slack_ratio = slack_ratios[signal_name]

            def node_cost_fn(n):
                b = node_base_cost.get(n, 0)
                h = node_history_cost.get(n, 0)
                p = node_present_sharing_cost.get(n, 0)
                c = (b + h) * (p + 1)
                return c * (1 - slack_ratio)

            def edge_cost_fn(e):
                (m, n) = e
                return edge_base_cost[(m, n)] * slack_ratio

            st = multi_via_router.route(Gs.get(signal_name, graph), terminals, node_cost_fn, edge_cost_fn)

            routing_trees[signal_name] = st

            # Include colliding nodes.
            nodes = set(chain(*(node_conflict.get(n, {}) for n in st.nodes)))
            nodes.update(st.nodes)

            for n in st.nodes:
                for o in node_conflict.get(n, {n}):
                    if o not in node_present_sharing_cost:
                        node_present_sharing_cost[o] = 0
                    node_present_sharing_cost[o] += 1

        # Detect node collisions

        # Get all nodes that are used for routing.
        all_routing_tree_nodes = set(chain(*(tree.nodes
                                             for tree in routing_trees.values())))

        # For each signal, get a set of nodes that are used by it.
        signal_nodes = (set(chain(*(node_conflict.get(n, {n}) for n in tree.nodes)))
                        for tree in routing_trees.values())

        # Count how many times a node is used.
        node_sharing = Counter(chain(*signal_nodes))

        # Find collisions
        node_collisions = [k for k, v in node_sharing.items() if v > 1 and k in all_routing_tree_nodes]

        has_collision = len(node_collisions) > 0

        if not has_collision:
            logger.info("Global routing done in %d iterations", j)
            break

        # node_history_cost = {n: c*0.9 for n,c in node_history_cost.items()}

        # Increment history cost
        for n in node_collisions:
            node_history_cost[n] = node_history_cost.get(n, 1) + 100

        # Update slack_ratios
        tree_weights = {signal_name: sum(edge_base_cost[e] for e in rt.edges())
                        for signal_name, rt in routing_trees.items()}
        max_weight = max(tree_weights.values())
        slack_ratios = {n: t / max_weight for n, t in tree_weights.items()}
        slack_ratios = {n: (s - 0.5) * 0.6 + 0.5 for n, s in slack_ratios.items()}

    return routing_trees


def test():
    logging.basicConfig(level=logging.INFO)
    import matplotlib.pyplot as plt
    from .signal_router import DijkstraRouter

    G = nx.Graph()

    num_x = 10
    num_y = 10
    x = range(0, num_x)
    y = range(0, num_y)

    # Store drawing positions of vertices. For plotting only.
    pos = {}

    # Construct mesh
    for name, (x, y) in enumerate(product(x, y)):
        G.add_node((x, y))
        pos[(x, y)] = (x, y)

        if x < num_x - 1 and not (1 <= y < 5 and x == 4):
            G.add_edge((x, y), (x + 1, y), weight=1, orientation='h')

        if y < num_y - 1:
            G.add_edge((x, y), (x, y + 1), weight=1, orientation='v')

    G.add_edge((8, 0), (9, 0), multi_via=2)

    # Draw mesh
    nx.draw_networkx(G, pos=pos, node_color='gray', node_size=8, edge_color='lightgray', hold=True)

    signals = {  # [terminals, ...]
        'a': [(0, 0), (8, 5), (7, 7), (6, 3)],
        'b': [(1, 1), (9, 0)],
        # [(3,3), (3,6)],
        # [(0,9), (9,0)],
        # [(0,1), (9,2)],
        # [(1,1), (8,9), (7,4)],
        # [(1,1), (2,0)],
        # [(4,1), (5,1), (7,3)],
        # [(1,3), (4,8)],
        # [(3,3), (6,5)],
        # [(1,2), (0,0), (0,4), (2,0), (2,4), (9,9), (5,5)],
        # [(1,1), (1,8)],
        # [(3,1), (3,8)],
        # [(9,0), (0,9), (5,5)],
    }

    colors = ['red', 'blue', 'green', 'orange', 'violet']
    router = PathFinderGraphRouter(DijkstraRouter())
    routing_trees = router.route(G, signals)
    edge_labels = {(a, b): "%.2f" % data.get('weight', 0) for (a, b, data) in G.edges(data=True)}

    for (name, signal), color in zip(signals.items(), colors):
        edges = list(routing_trees[name].edges)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=4, edge_color=color)

    nx.draw_networkx_nodes(G, pos, nodelist=list(chain(*signals.values())), node_size=32, node_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.draw()
    plt.show()
