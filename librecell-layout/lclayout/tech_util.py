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
from typing import Any, Dict, Tuple
import logging
import importlib.machinery
import types

logger = logging.getLogger(__name__)


def spacing_graph(min_spacing: Dict[Tuple[Any, Any], int]) -> nx.Graph:
    """
    Build a spacing rule graph by mapping the minimal spacing between layer a and layer b to an edge
    a-b in the graph with weight=min_spacing.
    """
    spacing_graph = nx.Graph()
    for (l1, l2), spacing in min_spacing.items():
        spacing_graph.add_edge(l1, l2, min_spacing=spacing)
    return spacing_graph


def load_tech_file(path, module_name='tech'):
    """
    Load a python module containing technology information.
    :param path:
    :param module_name:
    :return: Handle to the module.
    """
    logger.info('Loading tech file: %s', path)
    loader = importlib.machinery.SourceFileLoader('module_name', path)
    tech = types.ModuleType(loader.name)
    loader.exec_module(tech)
    return tech
