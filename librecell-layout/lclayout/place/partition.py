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
import logging

from itertools import count

from lccommon import net_util

logger = logging.getLogger(__name__)


def partition(ciruit_graph: nx.MultiGraph):
    """ Find subgraphs that are connected when ignoring supply nets.
    :param ciruit_graph: nx.MultiGraph
    :return:
    """

    logger.debug('Partitioning into connected subgraphs.')

    # Split supply nodes.
    g = nx.MultiGraph()
    cnt = count()

    node_reverse_mapping = dict()

    def replace_supply_node(n):
        if net_util.is_power_net(n):
            new = (n, next(cnt))
        else:
            new = (n, 0)
        node_reverse_mapping[new] = n
        return new

    for a, b, key, data in ciruit_graph.edges(keys=True, data=True):
        a = replace_supply_node(a)
        b = replace_supply_node(b)
        g.add_edge(a, b, key, **data)

    logger.debug('Connected components: %d', nx.number_connected_components(g))

    connected = (g.subgraph(c) for c in nx.connected_components(g))

    # Map node labels back to original labels.
    connected = [nx.relabel_nodes(c, node_reverse_mapping, copy=True) for c in connected]

    return connected
