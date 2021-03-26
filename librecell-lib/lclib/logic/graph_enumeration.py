#
# Copyright (c) 2019-2020 Thomas Kramer.
#
# This file is part of librecell 
# (see https://codeberg.org/tok/librecell).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# import sympy
# from sympy.logic.boolalg import to_cnf, to_dnf
# from sympy.logic import POSform, SOPform
# from sympy.utilities.lambdify import lambdify

from itertools import chain, product, count, combinations

import networkx as nx

from typing import Any, Dict, Tuple, List, Iterable

from lclayout.data_types import Transistor, ChannelType

import logging

logger = logging.getLogger(__name__)


def is_unique_under_isomorphism(g: nx.Graph, references: Iterable[nx.Graph]):
    def node_match(attr1, attr2):
        return attr1.get('type', None) == attr2.get('type', None)

    return not any((nx.is_isomorphic(g, ref, node_match=node_match) for ref in references))


def enum_graphs(max_path_len: int) -> List[Tuple[nx.Graph, Dict[Any, int]]]:
    assert max_path_len >= 0

    if max_path_len == 0:
        # Create graph with maximum path length = 0.
        g = nx.Graph()
        g.add_node("source", type="source")
        return [(nx.Graph(), {"source": 0})]
    else:
        prev = enum_graphs(max_path_len - 1)

        graphs = []

        for g, distances in prev:
            d_max = max(distances.values())
            d_max_nodes = [n for n, d in distances.items() if d == d_max]
            assert len(d_max_nodes) == 1
            d_max_node = d_max_nodes[0]

            non_dmax_nodes = [n for n, d in distances.items() if d < d_max]

            name_counter = count()
            for i in range(len(non_dmax_nodes) + 1):
                for comb in combinations(non_dmax_nodes, i):

                    g_new = g.copy()
                    distances_new = distances.copy()
                    new_node = "{}_{}".format(d_max + 1, next(name_counter))
                    g_new.add_node(new_node)
                    g_new.add_edge(new_node, d_max_node)
                    distances_new[new_node] = d_max + 1

                    for n in comb:
                        g_new.add_edge(new_node, n)

                    if is_unique_under_isomorphism(g_new, (g for g,_ in graphs)):
                        graphs.append((g_new, distances_new))

        return graphs


def test_enumerate_graphs():
    import matplotlib.pyplot as plt
    graphs = enum_graphs(3)

    print("Num graphs: ", len(graphs))

    # for g, dist in graphs:
    #     print(g)
    #     nx.draw_networkx(g, with_labels=True)
    #     plt.draw()
    #     plt.show()
