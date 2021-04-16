##
## Copyright (c) 2021 Thomas Kramer.
##
## This file is part of librecell-lib
## (see https://codeberg.org/tok/librecell/src/branch/master/librecell-lib).
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.
##

import itertools
import networkx as nx
from typing import Any, Dict, List, Iterable, Tuple, Set
from enum import Enum
import collections
import sympy
from sympy.logic import satisfiable, simplify_logic as sympy_simplify_logic
from sympy.logic import boolalg

from lclayout.data_types import ChannelType
import logging

# logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

"""
Recognize sequential cells based on the output of the `functional_abstraction.analyze_circuit_graph()` function.
"""



