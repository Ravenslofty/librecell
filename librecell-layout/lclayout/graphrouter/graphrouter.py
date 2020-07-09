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

from typing import Any, Dict, List, AbstractSet, Optional


class GraphRouter:

    def route(self,
              graph: nx.Graph,
              signals: Dict[Any, List[Any]],
              reserved_nodes: Optional[Dict[Any, AbstractSet[Any]]] = None,
              node_conflict: Optional[Dict[Any, AbstractSet[Any]]] = None,
              equivalent_nodes: Optional[Dict[Any, AbstractSet[Any]]] = None,
              is_virtual_node_fn=None
              ) -> Dict[Any, nx.Graph]:
        """

        :param graph: Routing graph.
        :param signals: Mapping of signal names to terminal nodes in the graph.
        :param reserved_nodes: Mapping of signal names to graph nodes that are reserved for this signal.
        :param node_conflict: Mapping of a node to other nodes that can not be used for routing at the same time.
        :param equivalent_nodes: For each node a set of nodes that are physically equivalent.
        :param is_virtual_node_fn: Function that returns True iff the argument is a virtual node.
        :return: Returns a dict mapping signal names to routing trees.
        """
        pass
