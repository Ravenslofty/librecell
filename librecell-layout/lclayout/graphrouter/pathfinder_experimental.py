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

from itertools import chain, count, combinations, product, takewhile
from collections import Counter
import numpy as np
from typing import Any, Dict, List, AbstractSet, Optional, Iterable, Tuple

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


def get_slack_ratios(routing_trees: Dict[Any, nx.Graph], edge_cost_fn) -> Dict[Any, float]:
    tree_weights = {signal_name: sum(edge_cost_fn(e) for e in rt.edges())
                    for signal_name, rt in routing_trees.items()}
    max_weight = max(tree_weights.values())
    slack_ratios = {n: t / max_weight for n, t in tree_weights.items()}
    # slack_ratios = {n: (s - 0.5) * 0.6 + 0.5 for n, s in slack_ratios.items()}

    return slack_ratios


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
    # multi_via_router = MultiViaRouter(detail_router, node_conflict)
    multi_via_router = detail_router

    #
    # max_iterations = 1000
    # for j in count():
    #
    #     if j >= max_iterations:
    #         raise Exception("Failed to route")
    #
    #     logger.info('Routing iteration %d' % j)

    # routing_order = sorted(signals.keys(), key=lambda i: slack_ratios[i], reverse=True)
    # node_present_sharing_cost.clear()

    past_lower_bounds = [1]

    def trial_route(node_history_cost: Dict[Any, float], slack: Dict[Any, float] = dict()):

        routing_order = sorted(signals.keys(), key=lambda i: (slack.get(i, 1), i), reverse=True)
        node_present_sharing_cost = dict()

        routing_trees = dict()

        #for signal_name in routing_order:
        for signal_name in signals.keys():
            terminals = signals[signal_name]

            #slack_ratio = slack.get(signal_name, 0.5)
            slack_ratio = 0.5

            def node_cost_fn(n):
                b = node_base_cost.get(n, 0)
                h = node_history_cost.get(n, 0)
                p = node_present_sharing_cost.get(n, 0)
                c = (b + h) * (p + 1)
                return c * (1 - slack_ratio)

            def edge_cost_fn(e):
                (m, n) = e
                return edge_base_cost[(m, n)] * slack_ratio

            # Compute spanning tree for current signal.
            st = multi_via_router.route(Gs.get(signal_name, graph),
                                        terminals,
                                        node_cost_fn,
                                        edge_cost_fn)

            routing_trees[signal_name] = st

            # # Increment present sharing cost of all nodes that are currently used by `st`,
            # # including colliding nodes.
            # nodes = set(chain(*(node_conflict.get(n, {}) for n in st.nodes)))
            # nodes.update(st.nodes)
            #
            # for n in st.nodes:
            #     for o in node_conflict.get(n, {n}):
            #         if o not in node_present_sharing_cost:
            #             node_present_sharing_cost[o] = 0
            #         node_present_sharing_cost[o] += 1

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

        return routing_trees, node_collisions

    routing_trees, node_collisions = trial_route(node_history_cost)

    for j in count():
        logger.info("Iteration {}".format(j))
        has_collision = len(node_collisions) > 0
        if not has_collision:
            logger.info("Global routing done in %d iterations", j)
            print("Iteration", j)
            break
        else:
            all_routing_tree_nodes = set(chain(*(tree.nodes
                                                 for tree in routing_trees.values())))

            def edge_cost_fn(edge):
                return edge_base_cost[edge]

            slack = get_slack_ratios(routing_trees, edge_cost_fn)

            def f(i: float) -> Tuple[bool, nx.Graph, List]:
                if node_collisions is not None and len(node_collisions) > 0:
                    _node_history_cost = node_history_cost.copy()
                    for n in set(node_collisions):  # & all_routing_tree_nodes:
                        _node_history_cost[n] = _node_history_cost.get(n, 0) + i
                else:
                    _node_history_cost = node_history_cost

                _routing_trees, _collisions = trial_route(_node_history_cost, slack)
                # Check if at least one routing tree has changed.
                is_different = any((
                    routing_trees[n].edges != _routing_trees[n].edges
                    for n in signals.keys()
                ))

                # for n in signals.keys():
                #     if routing_trees[n].edges != _routing_trees[n].edges:
                #         print(routing_trees[n].edges)
                #         print(_routing_trees[n].edges)
                #         break

                return is_different, _routing_trees, _collisions, _node_history_cost

            assert not f(0)[0], "Signal routing algorithm is not deterministic."

            # past_lower_bounds_sorted = list(sorted(past_lower_bounds))
            # initial_increment = past_lower_bounds[len(past_lower_bounds_sorted)//2]
            # print("initial increment", initial_increment)
            initial_increment = 10
            increments = (initial_increment * (i ** 2) for i in count())
            lower_bound = 0
            upper_bound = 0
            for i in increments:
                lower_bound = upper_bound
                upper_bound = i
                d, rt, col, hist_cost = f(i)
                if d:
                    upper_bound = i
                    break

            # past_lower_bounds.append(upper_bound)
            # if len(past_lower_bounds) > 10:
            #     past_lower_bounds = past_lower_bounds[1:]

            def bisect(lower, upper, tolerance=1) -> Tuple[float, Tuple[bool, nx.Graph, List]]:
                """
                Find smallest history cost increment such that a signal route is changed.
                Find smallest `i` such that `f` evaluates to `True`.
                :param lower:
                :param upper:
                :param tolerance:
                :return:
                """
                assert lower < upper

                middle = (lower + upper) / 2
                d, rt, col, hist_cost = f(middle)

                if (upper - lower) <= tolerance and d:
                    return middle, (d, rt, col, hist_cost)

                if d:
                    # Search in lower half
                    return bisect(lower, middle, tolerance=tolerance)
                else:
                    # Search in upper half
                    return bisect(middle, upper, tolerance=tolerance)

            increment, (d, rt, col, hist_cost) = bisect(lower_bound, upper_bound, tolerance=1)
            print(increment)
            assert d

            routing_trees = rt
            node_collisions = col
            node_history_cost = hist_cost

            # # Increment history cost
            # for n in set(node_collisions) & all_routing_tree_nodes:
            #     node_history_cost[n] = node_history_cost.get(n, 0) + increment

    return routing_trees


def test():
    logging.basicConfig(level=logging.INFO)
    import matplotlib.pyplot as plt
    from .signal_router import DijkstraRouter, ApproxSteinerTreeRouter

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
            G.add_edge((x, y), (x + 1, y), weight=10, orientation='h')

        if y < num_y - 1:
            G.add_edge((x, y), (x, y + 1), weight=10, orientation='v')

    G.add_edge((8, 0), (9, 0), multi_via=2)

    # Draw mesh
    nx.draw_networkx(G, pos=pos, node_color='gray', node_size=8, edge_color='lightgray', hold=True)

    signals = {  # [terminals, ...]
        'a': [(0, 0), (8, 5), (7, 7), (6, 3)],
        'b': [(1, 1), (9, 0)],
        'c': [(4, 1), (5, 1), (7, 3)],
        # 'd': [(1, 3), (4, 8)],
        # 'e': [(3,3), (6,5)],
        # [(3,3), (3,6)],
        # [(0,9), (9,0)],
        # [(0,1), (9,2)],
        # [(1,1), (8,9), (7,4)],
        # [(1,1), (2,0)],
        # 'f': [(1,2), (0,0), (0,4), (2,0), (2,4), (9,9), (5,5)],
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
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.draw()
    plt.show()
