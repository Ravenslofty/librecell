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
"""
Helper functions for finding minima and maxima.
"""

from typing import Iterable, List


def all_max(args: Iterable, key = None) -> List:
    """ Find all global maxima in `args`.
    :param args:
    :param key: Key function.
    :return: List of maxima.
    """

    if key is None:
        def key(x):
            return x

    maxima = []
    max_key = None

    for x in args:
        val = key(x)
        if max_key is None or val > max_key:
            maxima.clear()
            max_key = val
        if val == max_key:
            maxima.append(x)

    return maxima


def all_min(args: Iterable, key = None) -> List:
    """ Find all global minima in `args`.
    :param args:
    :param key: Key function.
    :return: List of minima.
    """

    if key is None:
        def key(x):
            return x

    minima = []
    min_key = None

    for x in args:
        val = key(x)
        if min_key is None or val < min_key:
            minima.clear()
            min_key = val
        if val == min_key:
            minima.append(x)

    return minima
