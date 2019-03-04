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
