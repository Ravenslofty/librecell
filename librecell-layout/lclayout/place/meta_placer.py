#
# Copyright 2019-2021 Thomas Kramer.
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
from ..data_types import *
from .place import TransistorPlacer

from . import euler_placer
from . import eulertours

import itertools
from typing import Iterable, List
import logging

logger = logging.getLogger(__name__)


class MetaTransistorPlacer(TransistorPlacer):
    """
    This placement engine is a wrapper around other engines.
    Based on simple heuristics it will call a suited engine.
    For example simple circuits will be placed with an algorithm that does not scale so well but is most
    accurate, big circuits will be placed with an algorithm that scales better but yields lower-quality results (in shorter time though).

    This involves extremely rough heuristics. Some more thorough theoretical analysis could make a big improvement.

    This is kind of a stupid fix because there's no good-enough placement engine.
    """

    def __init__(self):
        pass

    def get_placer(self, transistors: List[Transistor]) -> TransistorPlacer:
        """
        Heuristically find a suited placement engine.
        :param transistors:
        :return:
        """
        logger.debug('Estimate placement complexity.')

        # Try to find a heuristical measure for the placement complexity.
        # This is very vague.

        nmos = [t for t in transistors if t.channel_type == ChannelType.NMOS]
        pmos = [t for t in transistors if t.channel_type == ChannelType.PMOS]
        nmos_graph = euler_placer._transistors2graph(nmos)
        pmos_graph = euler_placer._transistors2graph(pmos)


        even_degree_graphs_n = eulertours.construct_even_degree_graphs(nmos_graph)
        num_even_degree_graphs_n = len(even_degree_graphs_n)
        logger.debug('Number of even-degree graphs (NMOS): %d', num_even_degree_graphs_n)
        if num_even_degree_graphs_n > 20:
            return euler_placer.HierarchicalPlacer()
        even_degree_graphs_p = eulertours.construct_even_degree_graphs(pmos_graph)
        num_even_degree_graphs_p = len(even_degree_graphs_p)
        logger.debug('Number of even-degree graphs (PMOS): %d', num_even_degree_graphs_p)
        if num_even_degree_graphs_p > 20:
            return euler_placer.HierarchicalPlacer()

        # Lazily construct all euler tours.
        max_tours = 200

        logger.debug('Find eulerian tours.')
        all_eulertours_n = list(chain(*(eulertours.find_all_euler_tours(g, limit=max_tours) for g in even_degree_graphs_n)))
        all_eulertours_p = list(chain(*(eulertours.find_all_euler_tours(g, limit=max_tours) for g in even_degree_graphs_p)))

        num_tours_n = len(all_eulertours_n)
        num_tours_p = len(all_eulertours_p)

        complexity = num_tours_n * num_tours_p

        logger.debug(f'Estimated placement complexity: {complexity}')

        if complexity < 400000:
            return euler_placer.EulerPlacer()
        else:
            return euler_placer.HierarchicalPlacer()

        return euler_placer.EulerPlacer()

    def place(self, transistors: Iterable[Transistor]) -> Cell:

        placer = self.get_placer(transistors)

        logger.info(f"Placement engine: {type(placer).__name__}")

        return placer.place(transistors)


