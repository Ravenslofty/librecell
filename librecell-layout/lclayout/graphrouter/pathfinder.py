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
import numpy as np
from .graphrouter import GraphRouter
from .signal_router import SignalRouter
from .multi_via_router import MultiViaRouter

from itertools import chain, count, combinations, product
from collections import Counter

from typing import Any, Dict, List, AbstractSet, Optional

import logging

logger = logging.getLogger(__name__)


class PathFinderGraphRouter(GraphRouter):

    def __init__(self,
                 detail_router: SignalRouter,
                 is_virtual_edge_fn=None):
        self.detail_router = detail_router

    def route(self,
              graph: nx.Graph,
              signals: Dict[Any, List[Any]],
              reserved_nodes: Optional[Dict] = None,
              node_conflict: Optional[Dict[Any, AbstractSet[Any]]] = None,
              equivalent_nodes: Optional[Dict[Any, AbstractSet[Any]]] = None,
              is_virtual_node_fn=None
              ) -> Dict[Any, nx.Graph]:
        return _route(self.detail_router,
                      graph,
                      signals=signals,
                      reserved_nodes=reserved_nodes,
                      node_conflict=node_conflict,
                      equivalent_nodes=equivalent_nodes,
                      is_virtual_node_fn=is_virtual_node_fn)


def _compute_tree_weight(routing_tree: nx.Graph, edge_cost_fn, is_virtual_edge_fn) -> float:
    """
    Compute the weight of a routing tree based on a edge cost function.
    :param routing_tree: A tree graph.
    :param edge_cost_fn: edge_cost((a,b)) = 'cost of edge from node a to node b'
    :param is_virtual_edge_fn: Tells wether an edge is 'virtual'. Edges adjacent to virtual nodes are not accounted in the tree weight.
    :return: The tree weight
    """
    tree_weights = sum(edge_cost_fn(e) for e in routing_tree.edges() if not is_virtual_edge_fn(e))
    return tree_weights


def _compute_tree_weights(routing_trees: Dict[Any, nx.Graph], edge_cost_fn, is_virtual_edge_fn) -> Dict[Any, float]:
    """
    Helper function which calls `_compute_tree_weight` for each element of a dictionary.
    :param routing_trees:
    :param edge_cost_fn:
    :param is_virtual_edge_fn:
    :return:
    """
    tree_weights = {signal_name: _compute_tree_weight(rt, edge_cost_fn, is_virtual_edge_fn)
                    for signal_name, rt in routing_trees.items()}
    return tree_weights


def _route(detail_router: SignalRouter,
           graph: nx.Graph,
           signals: Dict[Any, List[Any]],
           reserved_nodes: Optional[Dict] = None,
           node_conflict: Optional[Dict[Any, AbstractSet[Any]]] = None,
           equivalent_nodes: Optional[Dict[Any, AbstractSet[Any]]] = None,
           is_virtual_node_fn=None) -> Dict[Any, nx.Graph]:
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

    if is_virtual_node_fn is None:
        def is_virtual_node_fn(_):
            return False

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
        w = data['weight']
        edge_base_cost[(a, b)] = w
        edge_base_cost[(b, a)] = w

        # if a not in node_base_cost:
        #     node_base_cost[a] = 0
        #
        # if b not in node_base_cost:
        #     node_base_cost[b] = 0
        #
        # node_base_cost[a] += w//2
        # node_base_cost[b] += w//2

    def is_virtual_edge(e) -> bool:
        a, b = e
        return is_virtual_node_fn(a) or is_virtual_node_fn(b)

    # Pre-scaling: Normalize edge costs.
    edge_costs = [cost for edge, cost in edge_base_cost.items() if not is_virtual_edge(edge)]
    mean_edge_cost = np.mean(edge_costs)

    logger.debug('Mean edge cost (without virtual edges): {:.2f}'.format(mean_edge_cost))
    logger.debug('Pre-scaling factor for edge costs: 1/{:.2f}'.format(mean_edge_cost))

    edge_base_cost = {k: v / mean_edge_cost for k, v in edge_base_cost.items()}

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
                    Exception("Graph has been disconnected by removal of reserved nodes ({}).".format(net))

            Gs[net] = G2

    # TODO: just use detail_router and let caller decide whether to use MultiViaRouter or not.
    multi_via_router = MultiViaRouter(detail_router, node_conflict)
    # multi_via_router = detail_router  # Don't use multi via router.

    history_cost_weight = 1
    node_present_sharing_cost_increment = 10

    max_iterations = 1000

    for j in count():

        if j >= max_iterations:
            raise Exception("Failed to route")

        logger.info('Routing iteration %d' % j)

        routing_order = sorted(signals.keys(), key=lambda i: (slack_ratios[i], i), reverse=True)
        logger.debug('Routing order: {}'.format(routing_order))
        node_present_sharing_cost.clear()

        for signal_name in routing_order:
            terminals = signals[signal_name]

            slack_ratio = slack_ratios[signal_name]

            def node_cost_fn(n):
                b = node_base_cost.get(n, 0)
                h = node_history_cost.get(n, 0) * history_cost_weight
                p = node_present_sharing_cost.get(n, 0) * node_present_sharing_cost_increment
                c = (1 + b + h) * (p + 1)
                return c * (1 - slack_ratio)

            def edge_cost_fn(e):
                (m, n) = e
                b = edge_base_cost[(m, n)]
                return b * slack_ratio

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

        if equivalent_nodes:
            # Also mark nodes that are equivalent.
            node_collisions = set(chain(*(equivalent_nodes[n] for n in node_collisions)))

        has_collision = len(node_collisions) > 0

        # Compute weights of all signal trees.
        tree_weights = _compute_tree_weights(routing_trees, lambda e: edge_base_cost[e], is_virtual_edge)

        # Print routing tree weights.
        for signal_name, tree_weight in tree_weights.items():
            original_tree_weight = tree_weight * mean_edge_cost  # Invert pre-scaling.
            logger.debug('weight of {:>16}: {:.2f}'.format(signal_name, original_tree_weight))

        if not has_collision:
            logger.info("Global routing done in %d iterations", j)
            break

        # node_history_cost = {n: c*0.9 for n,c in node_history_cost.items()}

        # Increment history cost
        for n in node_collisions:
            node_history_cost[n] = node_history_cost.get(n, 0) + 1

        max_weight = max(tree_weights.values())
        slack_ratios = {n: t / max_weight for n, t in tree_weights.items()}
        slack_ratio_scaling = 0.1
        slack_ratios = {n: (s - 0.5) * slack_ratio_scaling + 0.5 for n, s in slack_ratios.items()}

    # Perform single-net optimizations.
    # For each net n: Rip up n while leaving the other nets untouched. Route n again.

    logger.info('Run single-net optimizations.')
    all_signal_names = signals.keys()
    for current_signal in all_signal_names:
        logger.debug('Single-net optimization: {}'.format(current_signal))
        other_signal_names = all_signal_names - current_signal

        other_routing_trees = {name: rt for name, rt in routing_trees.items() if name != current_signal}

        # Get the set of nodes occupied by the other signals.
        used_nodes = set(chain(*
                               (
                                   node_conflict.get(n, {n})
                                   for tree in other_routing_trees.values()
                                   for n in tree.nodes
                               )
                               )
                         )

        # Get the routing graph where all other signals are already routed.
        current_graph = Gs.get(current_signal, graph).copy()
        # Remove the nodes that are already in use by other signals.
        current_graph.remove_nodes_from(used_nodes)

        # Now the use the plain edge cost only (without history costs).
        def edge_cost_fn(e):
            (m, n) = e
            return edge_base_cost[(m, n)]

        # Find a route for the current signal.
        terminals = signals[current_signal]
        old_route = routing_trees[current_signal]
        new_route = multi_via_router.route(current_graph,
                                           terminals,
                                           node_cost_fn=lambda x: 0,
                                           edge_cost_fn=edge_cost_fn
                                           )

        old_weight = _compute_tree_weight(old_route, edge_cost_fn, is_virtual_edge)
        new_weight = _compute_tree_weight(new_route, edge_cost_fn, is_virtual_edge)
        logger.debug('Old weight for {}: {}'.format(current_signal, old_weight))
        logger.debug('New weight for {}: {}'.format(current_signal, new_weight))

        # Save the route.
        routing_trees[current_signal] = new_route

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

        w = 1

        if x < num_x - 1 and not (1 <= y < 5 and x == 4):
            G.add_edge((x, y), (x + 1, y), weight=w, orientation='h')

        if y < num_y - 1:
            G.add_edge((x, y), (x, y + 1), weight=w, orientation='v')

    G.add_edge((8, 0), (9, 0), multi_via=2)

    # Draw mesh
    nx.draw_networkx(G, pos=pos, node_color='gray', node_size=8, edge_color='lightgray', hold=True)

    signals = {  # [terminals, ...]
        'a': [(0, 0), (8, 5), (7, 7), (6, 3)],
        'b': [(1, 1), (9, 0)],
        'c': [(3, 3), (3, 6)],
        'd': [(4, 4)],
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
