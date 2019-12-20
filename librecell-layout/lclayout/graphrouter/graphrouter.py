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
