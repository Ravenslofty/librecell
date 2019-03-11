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

from pysmt.shortcuts import *
from pysmt.fnode import FNode
from pysmt.solvers.z3 import Z3Model, Z3Converter
from typing import Any, Dict, List
from itertools import product, chain
import logging

# Import z3 if package is available.
try:
    import z3

    _z3_available = True
except ImportError:
    _z3_available = False

logger = logging.getLogger(__name__)


class MIPSignalRouter(SignalRouter):

    def __init__(self):
        pass

    def route(self, G: nx.Graph,
              terminals: List[Any],
              node_cost_fn,
              edge_cost_fn
              ) -> nx.Graph:
        return min_steiner_tree(
            G,
            [terminals],
            node_cost_fn,
            edge_cost_fn,
        )[0]


class MIPGraphRouter():

    def route(self, G: nx.Graph,
              terminals: List[Any],
              node_cost_fn,
              edge_cost_fn
              ) -> nx.Graph:
        return min_steiner_tree(
            G,
            terminals,
            node_cost_fn,
            edge_cost_fn,
        )


def min_steiner_tree(
        G: nx.Graph,
        signals: List[List[Any]],
        node_cost_fn,
        edge_cost_fn,
) -> nx.Graph:
    """
    Find a minimum weight Steiner tree which connects all `terminals` in `G`.

    :param G: Graph.
    :param terminals: List of terminals to be connected.

    References:
    Sunil Chopra and Chih-Yang Tsai, Polyhedral approaches for the Steiner Tree on graphs, 2000.
    """

    optimize = False
    solver_name = 'z3'
    debug = False

    if optimize:
        assert _z3_available, "Need z3-solver for optimizations."

    if optimize and solver_name != 'z3':
        logger.warning("Can not use {} for optimizations. Switching to z3.".format(solver_name))
        solver_name = 'z3'

    # Get solver instance.
    if debug:
        logger.debug("Use UnsatCoreSolver.")
        solver = UnsatCoreSolver(name=solver_name)
    else:
        solver = Solver(name=solver_name)

    # Create optimizer instances if needed.
    if optimize:
        optimizer = z3.Optimize()
        converter = solver.converter
        logger.info("SMT optimizer: z3")
    else:
        optimizer = None
        converter = None
        logger.info("SMT solver: {}".format(type(solver).__name__))

    def add_assertion(assertion: FNode, **kwargs):
        """
        Add assertion to solver and optimizer.
        :param assertion:
        :param kwargs:
        :return:
        """

        assert isinstance(assertion, FNode)
        solver.add_assertion(assertion, **kwargs)
        if optimizer:
            optimizer.add(
                converter.convert(assertion)
            )

    def minimize(objective: FNode):
        """
        Add minimization objective to optimizer.
        :param objective:
        :return:
        """
        if optimizer:
            optimizer.minimize(
                converter.convert(objective)
            )

    def create_constraints(source, sinks) -> Dict[Any, FNode]:
        """
        Create constraints for steiner tree of a single signal.
        :param source:
        :param sinks:
        :return:
        """
        # Edge choice variables.
        x = {(i, j): FreshSymbol(INT, template='x%d')
             for i, j in G.edges()
             }

        # Flow variables
        f = dict()
        f.update({(k, i, j): FreshSymbol(INT, template='f%d')
                  for k, (i, j) in product(sinks, G.edges)
                  })
        f.update({(k, j, i): FreshSymbol(INT, template='f%d')
                  for k, (i, j) in product(sinks, G.edges)
                  })

        y = dict()
        y.update({(i, j): FreshSymbol(INT, template='y%d') for i, j in G.edges})
        y.update({(j, i): FreshSymbol(INT, template='y%d') for i, j in G.edges})

        # Constraint (1)
        for k in sinks:
            for j in G.nodes:
                left = Int(0) + sum((f[(k, i, j)] - f[(k, j, i)] for i in G.neighbors(j)))

                if j == source:
                    right = -1
                elif j == k:
                    right = 1
                else:
                    right = 0

                add_assertion(left.Equals(Int(right)))

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
                Equals(y_ij + y_ji, x_ij)
            )

        # f_kij must be >= 0
        for f_kij in f.values():
            add_assertion(f_kij >= 0)

        return x

    def create_node_usage_indicator(edge_usage: Dict[Any, FNode]) -> Dict[Any, FNode]:
        """
        Create variables indicating if a node is used based on usage of edges.
        :param edge_usage:
        :return:
        """
        # Introduce variable for indicating node usage.
        node_used = {n: FreshSymbol(INT, template='node_used_%d')
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

    total_cost = Int(0)

    xs = []
    if len(signals) > 1:
        node_sharing = {n: 0 for n in G.nodes}
    else:
        node_sharing = dict()

    for terminals in signals:
        source = terminals[0]
        sinks = terminals[1:]
        x = create_constraints(source, sinks)
        xs.append(x)

        node_used = create_node_usage_indicator(x)

        # Count by how many signals a node is used.
        if len(signals) > 1:
            for n, var in node_used.items():
                node_sharing[n] += var

        if node_cost_fn is not None:
            # Node weights.

            node_weight = sum(
                Int(node_cost_fn(n)) * node_used[n]
                for n in G.nodes
            )

            total_cost += node_weight

        if edge_cost_fn is not None:
            edge_weight = sum(
                # Int(data.get('weight', 0)) * x[(i, j)]
                Int(edge_cost_fn((i, j))) * x[(i, j)]
                for i, j, data in G.edges(data=True)
            )

            total_cost += edge_weight

    # Distinct nodes constraint.
    # Assert that a node cannot be shared among signals.
    for n, num_signals in node_sharing.items():
        add_assertion(num_signals <= 1)

    # Minimize sum of edge weights.
    minimize(total_cost)

    if optimizer:
        logger.info('Invoke SMT optimizer')
        sat = optimizer.check()
        logger.info('SMT optimize result: %s', 'SAT' if sat else 'UNSAT')
    else:
        logger.info('Invoke SMT solver')
        sat = solver.check_sat()
        logger.info('SMT result: %s', 'SAT' if sat else 'UNSAT')

    if sat:
        if optimizer:
            model = Z3Model(solver.environment, optimizer.model())
        else:
            model = solver.get_model()

        # Map solution back to a routing graph.
        routing_trees = []
        for x in xs:
            routing_tree = nx.Graph()
            for i, j in G.edges:
                x_ij = x[(i, j)]
                value = model.get_py_value(x_ij)
                if value == 1:
                    routing_tree.add_edge(i, j)
            routing_trees.append(routing_tree)

        return routing_trees

    else:
        msg = 'UNSAT: Constraints not satisfiable.'
        logger.error(msg)
        if debug:
            core = solver.get_named_unsat_core()
            logger.info('unsat core: %s', list(core.keys()))
            logger.debug('unsat core: %s', core)

        raise Exception(msg)


def test_mip_route():
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

    signals = [  # [terminals, ...]
        [(0, 0), (8, 5), (7, 7), (6, 3)],
        [(1, 1), (9, 0)],
        #[(3, 3), (3, 6)],
        #[(0, 9), (9, 0)],
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
    ]

    routing_trees = min_steiner_tree(G, signals,
                                     node_cost_fn=None,
                                     edge_cost_fn=lambda e: 1)

    colors = ['red', 'blue', 'green', 'orange', 'violet']
    # routing_trees = route_hv(DijkstraRouter(), G, signals, orientation_change_penalty=10)

    # edge_labels = {(a, b): "%.2f" % data.get('weight', 0) for (a, b, data) in G.edges(data=True)}

    # nx.draw_networkx_edges(G, pos, edgelist=routing_tree.edges, width=4, edge_color=colors[0])

    for i, signal in enumerate(signals):
        edges = list(routing_trees[i].edges)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=4, edge_color=colors[i])

    nx.draw_networkx_nodes(G, pos, nodelist=list(chain(*signals)), node_size=32, node_color='black')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.draw()
    plt.show()
