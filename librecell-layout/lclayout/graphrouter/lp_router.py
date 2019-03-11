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
from .signal_router import SignalRouter

from typing import Any, Dict, List, Optional, Union, AbstractSet
from itertools import product, chain, count
import logging

# Import GLPK binding
from pulp import *
from pulp.constants import LpStatus

logger = logging.getLogger(__name__)


class LPSignalRouter(SignalRouter):

    def __init__(self):
        pass

    def route(self, G: nx.Graph,
              terminals: List[Any],
              node_cost_fn,
              edge_cost_fn
              ) -> nx.Graph:
        return min_steiner_trees(
            G,
            {0: terminals},
            node_cost_fn,
            edge_cost_fn,
        )[0]


class LPGraphRouter():

    def route(self, G: nx.Graph,
              signals: Dict[Any, List[Any]],
              reserved_nodes: Optional[Dict] = None,
              node_conflict: Optional[Dict[Any, AbstractSet[Any]]] = None,
              node_cost_fn=None,
              edge_cost_fn=None
              ) -> Dict[Any, nx.Graph]:
        assert isinstance(signals, dict)

        if edge_cost_fn is None:
            edge_cost_fn = lambda x: G[x[0]][x[1]]['weight']

        routing_trees = min_steiner_trees(
            G,
            signals,
            node_cost_fn=node_cost_fn,
            edge_cost_fn=edge_cost_fn,
            reserved_nodes=reserved_nodes,
            node_conflict=node_conflict
        )

        return routing_trees


def min_steiner_trees(
        G: nx.Graph,
        signals: Dict[Any, AbstractSet[Any]],
        node_cost_fn,
        edge_cost_fn,
        reserved_nodes: Optional[Dict[Any, AbstractSet[Any]]] = None,
        node_conflict: Optional[Dict[Any, AbstractSet[Any]]] = None
) -> Dict[Any, nx.Graph]:
    """
    Find a minimum weight Steiner trees which connects all `terminals` of all `signals` in `G`
    but without creating shorts between the signals.

    :param G: Graph.
    :param terminals: List of terminals to be connected.

    References:
    Sunil Chopra and Chih-Yang Tsai, Polyhedral approaches for the Steiner Tree on graphs, 2000.
    """

    if node_conflict is None:
        node_conflict = dict()

    if reserved_nodes is None:
        reserved_nodes = dict()

    # Get solver instance.
    problem = LpProblem("steiner_tree", LpMinimize)

    def add_assertion(assertion: LpConstraint, problem=problem):
        """
        Add assertion to solver.
        :param assertion:
        :param kwargs:
        :return:
        """
        problem += assertion

    _counter = count()

    def fresh_variable(template: str = 'var%d') -> LpVariable:
        """
        Create fresh integer variable.
        :param template:
        :return:
        """
        return LpVariable(template % next(_counter), cat='Continuous')

    def fresh_int_variable(template: str = 'var%d') -> LpVariable:
        """
        Create fresh integer variable.
        :param template:
        :return:
        """
        return LpVariable(template % next(_counter), cat='Integer')

    # def create_constraints_multi_commodity_flow(problem: LpProblem, source, sinks) -> Dict[Any, LpVariable]:
    #     """
    #     Create constraints for steiner tree of a single signal.
    #
    #     Experiment: model a flow from one source to multiple sinks. This has the advantage of using less
    #     variables and less constraints but shows to be slower in practice that modelling multiple
    #     single-commodity flows.
    #
    #     :param source:
    #     :param sinks:
    #     :return:
    #     """
    #     # Edge choice variables.
    #     x = {(i, j): fresh_int_variable(template='x%d')
    #          for i, j in G.edges()
    #          }
    # 
    #     # Flow variables
    #     f = dict()
    #     f.update({(i, j): fresh_variable(template='f%d')
    #               for (i, j) in G.edges
    #               })
    #     f.update({(j, i): fresh_variable(template='f%d')
    #               for (i, j) in G.edges
    #               })
    #
    #     y = dict()
    #     y.update({(i, j): fresh_variable(template='y%d') for i, j in G.edges})
    #     y.update({(j, i): fresh_variable(template='y%d') for i, j in G.edges})
    #
    #     # Constraint (1)
    #     for j in G.nodes:
    #         left = sum((f[(i, j)] - f[(j, i)] for i in G.neighbors(j)))
    #
    #         if j == source:
    #             right = -1 * len(sinks)
    #         elif j in sinks:
    #             right = 1
    #         else:
    #             right = 0
    #
    #         add_assertion(left == right)
    #
    #     # Constraint (3)
    #     for (i, j) in G.edges:
    #         scale = 1 / len(sinks)
    #         add_assertion(f[(i, j)] * scale <= y[(i, j)])
    #         add_assertion(f[(j, i)] * scale <= y[(j, i)])
    #
    #     # Constraint (4)
    #     for i, j in G.edges:
    #         y_ij = y[(i, j)]
    #         y_ji = y[(j, i)]
    #         x_ij = x[(i, j)]
    #         add_assertion(
    #             y_ij + y_ji == x_ij
    #         )
    #
    #     # f_kij must be >= 0
    #     for f_ij in f.values():
    #         add_assertion(f_ij >= 0)
    #
    #     return x

    def create_constraints_single_commodity_flow(problem: LpProblem, source, sinks) -> Dict[Any, LpVariable]:
        """
        Create constraints for steiner tree of a single signal.
        :param source:
        :param sinks:
        :return:
        """
        # Edge choice variables.
        x = {(i, j): fresh_int_variable(template='x%d')
             for i, j in G.edges()
             }

        # Flow variables
        f = dict()
        f.update({(k, i, j): fresh_variable(template='f%d')
                  for k, (i, j) in product(sinks, G.edges)
                  })
        f.update({(k, j, i): fresh_variable(template='f%d')
                  for k, (i, j) in product(sinks, G.edges)
                  })

        y = dict()
        y.update({(i, j): fresh_variable(template='y%d') for i, j in G.edges})
        y.update({(j, i): fresh_variable(template='y%d') for i, j in G.edges})

        # Constraint (1)
        # For each source/sink pair there must be a flow of value 1 from source to sink k.
        # The net flow for non-sink and non-source nodes must be 0.
        # The net flow for the source node must be -1 and for the sink node +1.
        for k in sinks:
            for j in G.nodes:
                left = sum((f[(k, i, j)] - f[(k, j, i)] for i in G.neighbors(j)))

                if j == source:
                    right = -1
                elif j == k:
                    right = 1
                else:
                    right = 0

                add_assertion(left == right)

        # Constraint (3)
        for k, (i, j) in product(sinks, G.edges):
            add_assertion(f[(k, i, j)] <= y[(i, j)])
            add_assertion(f[(k, j, i)] <= y[(j, i)])

        # Constraint (4)
        for i, j in G.edges:
            y_ij = y[(i, j)]
            y_ji = y[(j, i)]
            x_ij = x[(i, j)]
            add_assertion(
                y_ij + y_ji == x_ij
            )

        # f_kij must be >= 0
        for f_kij in f.values():
            add_assertion(f_kij >= 0)

        return x

    def create_node_usage_indicator(edge_usage: Dict[Any, LpVariable]) -> Dict[Any, LpVariable]:
        """
        Create variables indicating if a node is used based on usage of edges.
        :param edge_usage:
        :return:
        """
        # Introduce variable for indicating node usage.
        node_used = {n: fresh_variable(template='node_used_%d')
                     for n in G.nodes}
        # node_used can be either 0 or 1
        for n, var in node_used.items():
            add_assertion(var >= 0)
            add_assertion(var <= 1)
        # node_used must be 1 if an adjacent edge is used.
        for i, j in G.edges:
            add_assertion(
                node_used[i] >= edge_usage[(i, j)]
            )
            add_assertion(
                node_used[j] >= edge_usage[(i, j)]
            )

        return node_used

    total_cost = 0

    used_edges_by_signal = dict()
    # `node_sharing` holds for each node the number of signals that are using the node.
    if len(signals) > 1:
        node_sharing = {n: 0 for n in G.nodes}
    else:
        node_sharing = dict()

    used_nodes_by_signal = dict()

    # For each signal:
    # * create connectivity constraint
    for signal_name, terminals in signals.items():
        source = terminals[0]
        sinks = terminals[1:]
        used_edges = create_constraints_single_commodity_flow(problem, source, sinks)
        # used_edges = create_constraints_multi_commodity_flow(problem, source, sinks)
        used_edges_by_signal[signal_name] = used_edges

        nodes_used = create_node_usage_indicator(used_edges)

        # Add constraint for node conflicts between different signals.
        # If node n1 is used, then mark all conflicting nodes also as used.
        for n1, conflicts in node_conflict.items():
            for n2 in conflicts:
                problem += nodes_used[n2] >= nodes_used[n1]
                problem += nodes_used[n1] >= nodes_used[n2]

        used_nodes_by_signal[signal_name] = nodes_used

        # Count by how many signals a node is used.
        if len(signals) > 1:
            for n, var in nodes_used.items():
                node_sharing[n] += var

        if node_cost_fn is not None:
            # Node weights.

            node_weight = sum(
                node_cost_fn(n) * nodes_used[n]
                for n in G.nodes
            )

            total_cost += node_weight

        if edge_cost_fn is not None:
            edge_weight = sum(
                edge_cost_fn((i, j)) * used_edges[(i, j)]
                for i, j, data in G.edges(data=True)
            )

            total_cost += edge_weight

    # Distinct nodes constraint.
    # Assert that a node cannot be shared among signals.
    for n, num_signals in node_sharing.items():
        add_assertion(num_signals <= 1)

    # Add constraints for reserved nodes.
    all_nets = set(signals.keys())
    for net_name, reserved in reserved_nodes.items():
        other_nets = all_nets - {net_name}
        # If a node is reserved for some signal, then the other signals can't use it.
        for net in other_nets:
            for r in reserved:
                problem += used_nodes_by_signal[net][r] == 0

    # Minimize sum of edge weights.
    problem.setObjective(total_cost)

    logger.info('Invoke LP solver')
    problem.solve()
    logger.info('LP solver result: %s', LpStatus[problem.status])

    if problem.status == 1:
        # Map solution back to a routing graph.
        routing_trees = dict()
        for net_name, used_edges in used_edges_by_signal.items():
            routing_tree = nx.Graph()
            for i, j in G.edges:
                x_ij = used_edges[(i, j)]
                value = x_ij.varValue
                if value > 0:
                    routing_tree.add_edge(i, j)
            routing_trees[net_name] = routing_tree

        return routing_trees

    else:
        msg = 'Problem not solved: {}'.format(LpStatus[problem.status])
        logger.error(msg)
        raise Exception(msg)


def test_lp_route():
    logging.basicConfig(level=logging.INFO)
    import matplotlib.pyplot as plt

    G = nx.Graph()

    num_x = 10
    num_y = 10
    x = range(0, num_x)
    y = range(0, num_y)

    # Store drawing positions of vertices. For plotting only.
    pos = {}

    # Construct mesh
    for i, (x, y) in enumerate(product(x, y)):
        G.add_node((x, y))
        pos[(x, y)] = (x, y)

        if x < num_x - 1 and not (1 <= y < 5 and x == 4):
            G.add_edge((x, y), (x + 1, y), weight=1, orientation='h')

        if y < num_y - 1:
            G.add_edge((x, y), (x, y + 1), weight=1, orientation='v')

    # Draw mesh
    nx.draw_networkx(G, pos=pos, node_color='gray', node_size=8, edge_color='lightgray', hold=True)

    signals = {  # [terminals, ...]
        'a': [(0, 0), (8, 5), (7, 7), (6, 3)],
        'b': [(1, 1), (9, 0)],
        # [(3, 3), (3, 6)],
        # [(0, 9), (9, 0)],
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

    conflicts = {(1, 0): {(0, 1)}}

    reserved_nodes = {
        'a': {(3, 0), (0, 1)}
    }

    routing_trees = min_steiner_trees(G, signals,
                                      node_cost_fn=None,
                                      edge_cost_fn=lambda e: 1,
                                      node_conflict=conflicts,
                                      reserved_nodes=reserved_nodes)

    colors = ['red', 'blue', 'green', 'orange', 'violet']
    # routing_trees = route_hv(DijkstraRouter(), G, signals, orientation_change_penalty=10)

    # edge_labels = {(a, b): "%.2f" % data.get('weight', 0) for (a, b, data) in G.edges(data=True)}

    # nx.draw_networkx_edges(G, pos, edgelist=routing_tree.edges, width=4, edge_color=colors[0])

    for i, (signal_name, signal) in enumerate(signals.items()):
        edges = list(routing_trees[signal_name].edges)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=4, edge_color=colors[i])

    nx.draw_networkx_nodes(G, pos, nodelist=list(chain(*signals.values())), node_size=32, node_color='black')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.draw()
    plt.show()
