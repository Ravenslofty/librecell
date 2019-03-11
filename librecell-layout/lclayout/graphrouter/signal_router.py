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

from itertools import product, tee, count
from heapq import heappush, heappop

from typing import Iterable, Mapping, TypeVar, AbstractSet, Any, List

from .. import extrema

from networkx.algorithms.approximation.steinertree import steiner_tree


def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


class SignalRouter:

    def route(self, G: nx.Graph,
              terminals: List,
              node_cost_fn,
              edge_cost_fn
              ) -> nx.Graph:
        pass


class AStarRouter(SignalRouter):

    def __init__(self, heuristic_fn=None):

        if heuristic_fn is not None:
            self.heuristic_fn = heuristic_fn
        else:
            self.heuristic_fn = lambda n: 0

    def route(self, G: nx.Graph,
              terminals: List,
              node_cost_fn,
              edge_cost_fn
              ) -> nx.Graph:
        return spanning_subtree(
            G,
            terminals,
            node_cost_fn,
            edge_cost_fn,
            heuristic_fn=self.heuristic_fn,
        )


class ApproxSteinerTreeRouter(SignalRouter):

    def route(self, G: nx.Graph,
              terminals: List,
              node_cost_fn,
              edge_cost_fn
              ) -> nx.Graph:

        ccs = [G.subgraph(c) for c in nx.connected_components(G)]

        subgraph = None
        for cc in ccs:
            if all(t in cc for t in terminals):
                subgraph = cc
                break
        assert subgraph is not None
        G2 = subgraph

        for a, b in G2.edges:
            w = 0
            if node_cost_fn is not None:
                w += node_cost_fn(a) + node_cost_fn(b)
            if edge_cost_fn is not None:
                w += edge_cost_fn((a, b))
            G2[a][b]['weight'] = w

        return steiner_tree(G2, terminals, weight='weight')


class DijkstraRouter(SignalRouter):

    def __init__(self):
        pass

    def route(self, G: nx.Graph,
              terminals: List[Any],
              node_cost_fn,
              edge_cost_fn
              ) -> nx.Graph:
        return spanning_subtree(
            G,
            terminals,
            node_cost_fn,
            edge_cost_fn,
        )


def spanning_subtree(
        G: nx.Graph,
        terminals: List,
        node_cost_fn,
        edge_cost_fn,
        heuristic_fn=None,
) -> nx.Graph:
    """ Creates a spanning subtree connecting all terminals.
    Assumes a connected graph G.

    Based on PathFinder and Prim's algorithm.

    Parameters
    ----------
    G: nx.Graph
            Routing graph
    terminal: List of terminal nodes
            Terminal nodes of signals to be connected.
    node_cost_fn: node -> number
    edge_cost_fn  node, node -> number
    heuristic_fn: function source, target -> number
            See `dijkstra_traverse` function
    """

    S = nx.Graph()
    #
    # center, center_dist = absolute_1_center(
    #     G,
    #     terminals,
    #     node_cost_fn,
    #     edge_cost_fn
    # )
    #
    # source = center
    source = terminals[0]

    sinks = set(terminals)
    sinks -= {source}

    S.add_node(source)

    while sinks:
        # Initialize priority queue with current routing tree at cost 0.

        sources = set(S.nodes)

        path = shortest_path(G,
                             sources,
                             sinks,
                             node_cost_fn,
                             edge_cost_fn,
                             heuristic_fn=heuristic_fn
                             )

        nx.add_path(S, path)
        end = path[-1]
        sinks.remove(end)

    return S


def absolute_1_center(
        G,
        terminals,
        node_cost_fn,
        edge_cost_fn
):
    # Create distance maps for all terminals.
    distances = {
        t: dijkstra(
            G,
            [t],
            node_cost_fn,
            edge_cost_fn
        )
        for t in terminals}

    distances_by_node = dict()
    for t, dists in distances.items():
        for n, dist in dists.items():
            distances_by_node.setdefault(n, []).append(dist)

    max_distances = {n: max(dists) for n, dists in distances_by_node.items()}
    # Use sum of squared distances as metric.
    # max_distances = {n: sum(map(lambda x: x ** 2, dists)) for n, dists in distances_by_node.items()}

    min_max_distance = min(max_distances.items(), key=lambda x: x[1])

    return min_max_distance


def dijkstra(
        G,
        sources,
        node_cost_fn,
        edge_cost_fn,
        terminals=None,
        closest_terminal_only=False,
        heuristic_fn=None
):
    """ Create a distance map from all nodes to the source.
    If `terminals` is not None, then the search stops as soon as all terminals are found.

    Parameters
    ----------
    sources: Nodes to start the search from.
    terminals: Destinations.
    slack_ration: Tradeoff between base cost and history/sharing cost. If set to 1, only the base cost is taken into account.
    max_paths: Maximum number of paths to find. If set to 1 only the shortest path from a source to a terminal node will be returned.
    closest_terminal_only: If set to True, the search will terminate as soon as the first terminal is found.
    heuristic_fn: See `dijkstra_traverse` function
    """

    sinks = None
    if terminals:
        sinks = set(terminals)

    def node_handler_fn(n):
        if closest_terminal_only:
            return False

        if sinks:
            sinks.remove(n)
            return len(sinks) > 0
        else:
            return True

    if terminals is None:
        def multi_target_heuristic(source):
            return 0
    else:
        def multi_target_heuristic(source):
            return min((heuristic_fn(source, t) for t in terminals))

    result = dijkstra_traverse(
        G,
        sources,
        node_cost_fn,
        edge_cost_fn,
        node_handler_fn,
        heuristic_fn=multi_target_heuristic
    )

    return result


def dijkstra_traverse(
        G,
        sources,
        node_cost_fn,
        edge_cost_fn,
        node_handler_fn,
        heuristic_fn=None
):
    """ Create a distance map from all nodes to the source.
    Nodes are visited in increasing distance order and passed to `node_handler_fn`. The search is continued
    as long as `node_handler_fn` returns `True` and there are unvisited nodes left.

    Parameters
    ----------
    sources: Nodes to start the search from.
    node_cost_fn: Node cost function. node -> cost
    edge_cost_fn: Edge cost function. (node, node) -> cost
    node_handler_fn: A function Node -> Bool
            Each node will be passed to this function in increasing distance order. The search will be aborted if the handler returns `False`.

    heuristic_fn: Source -> Estimated cost to reach target.
            Heuristic function to estimate the cost from a node to a target.
            Shortest paths are found as long as the heuristic does not overestimate costs.
    """

    if heuristic_fn is None:
        def h(n):
            return 0

        heuristic_fn = h

    class PQElement:
        def __init__(self, priority, value):
            self.priority = priority
            self.value = value

        def __cmp__(self, other):
            if self.priority < other.priority:
                return -1
            elif self.priority > other.priority:
                return 1
            return 0

        def __lt__(self, other):
            return self.priority < other.priority

        def __gt__(self, other):
            return self.priority > other.priority

        def as_tuple(self):
            return self.priority, self.value

    # Initialize priority queue with source node at cost 0.
    pq = [PQElement(node_cost_fn(n), n) for n in sources]

    c = count()
    result = dict()
    # result = {n: node_cost_fn(n) for n in sources}
    visited = set()
    enqueued = dict()
    n_nodes = len(G)
    # Storage for trace.
    prev_node = dict()
    while pq:
        # Continue search from lowest-cost node.
        cost_m, m = heappop(pq).as_tuple()

        if m in visited:
            continue
        visited.add(m)

        result[m] = cost_m

        if node_handler_fn is not None:
            if not node_handler_fn(m):
                break

        if len(visited) == n_nodes:
            break

        # Loop over fanout nodes
        for edge in G.edges(m, data=True):
            _, n, data = edge

            if n in visited:
                continue

            effective_cost = edge_cost_fn((m, n)) + node_cost_fn(n)

            # Get previous node if any.
            previous = prev_node.get(n, None)

            cost_n = cost_m + effective_cost

            # old_cost_h = enqueued.get(n, None)
            # if old_cost_h is not None:
            #     # h was already computed, no need to recompute it again.
            #     cost_old, h = old_cost_h
            #
            #     if cost_old <= cost_n:
            #         # We already have a better candidate.
            #         continue
            # else:
            #     h = heuristic_fn(n)

            h = heuristic_fn(n)
            heappush(pq, PQElement(cost_n + h, n))
            # Cache h
            # enqueued[n] = cost_n, h

            # Remember node if it is augmenting the path.
            if previous is not None:
                prev_node[n] = min(previous, (cost_n, m), key=lambda x: x[0])
            else:
                prev_node[n] = (cost_n, m)

    return result


def shortest_path(
        G,
        sources,
        terminals,
        node_cost_fn,
        edge_cost_fn,
        heuristic_fn=None
):
    """ Finds the shortest path from the source to one of the terminals.

    Based on PathFinder and Prim's algorithm.

    Parameters
    ----------
    sources: Nodes to start the search from.
    terminals: Destinations.
    slack_ration: Tradeoff between base cost and history/sharing cost. If set to 1, only the base cost is taken into account.
    max_paths: Maximum number of paths to find. If set to 1 only the shortest path from a source to a terminal node will be returned.
    """
    sinks = None
    if terminals:
        sinks = set(terminals)

    if heuristic_fn is None:
        multi_target_heuristic = None
    else:
        if terminals:
            def multi_target_heuristic(source):
                return min((heuristic_fn(source, t) for t in terminals))
        else:
            def multi_target_heuristic(source):
                return 0

    closest_terminal = []

    def node_handler_fn(n):
        if sinks:
            # Continue as long as n is not a sink.
            if n in sinks:
                closest_terminal.append(n)
                return False
            return True
        else:
            return True

    distance_map = dijkstra_traverse(
        G,
        sources,
        node_cost_fn,
        edge_cost_fn,
        node_handler_fn,
        heuristic_fn=multi_target_heuristic
    )

    # Find path from closest terminal to a sink.

    assert len(closest_terminal) == 1
    closest_terminal = closest_terminal[0]
    assert closest_terminal in sinks

    result = trace_back(G, distance_map, closest_terminal, sources)

    return result


N = TypeVar('N')


def trace_back(G: nx.Graph, distance_map: Mapping[N, int], source: N, targets: AbstractSet[N]):
    """ Find the shortest path from `source` to one of the targets
    by tracing back based on a distance map.

    Parameters
    ----------
    G: nx.Graph
    distance_map: Distance to the source.
    source: Start node.
    targets: End nodes.
    """

    targets = set(targets)

    current = source

    trace = [current]

    while current not in targets:
        neighbors = G.neighbors(current)
        neighbors = filter(lambda n: n in distance_map, neighbors)

        # neighbors = sorted(neighbors)
        closest_nodes = extrema.all_min(neighbors, key=lambda n: distance_map[n])
        # TODO: Which node is next if there are multiple closest nodes?
        next_node = closest_nodes[0]
        trace.append(next_node)
        current = next_node

    trace.reverse()

    return trace
