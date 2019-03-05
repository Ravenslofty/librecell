import networkx as nx

from typing import Any, Dict, List, AbstractSet, Optional


class GraphRouter:

    def route(self,
              graph: nx.Graph,
              signals: Dict[Any, List[Any]],
              reserved_nodes: Optional[Dict] = None,
              node_conflict: Optional[Dict[Any, AbstractSet[Any]]] = None
              # node_cost_fn,
              # edge_cost_fn
              ) -> Dict[Any, nx.Graph]:
        pass
