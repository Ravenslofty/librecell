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
from pysmt.shortcuts import *
import re

from pysmt.environment import get_env

"""
Collection of theorem proofs.
Just a scratch pad. This code is not directly used in lclayout. 
"""


def proof(f):
    return is_valid(f)


def is_valid(f):
    """ Proof the validity of a formula.
    """
    s = Solver()
    s.add_assertion(Not(f))

    has_counter_example = s.check_sat()

    if has_counter_example:
        m = s.get_model()
        print(m)

    return not has_counter_example


def is_equal(f1, f2):
    return proof(
        Iff(f1, f2)
    )


def symbols(names, dtype):
    return [Symbol(n, dtype) for n in re.split(',\s*|\s+', names)]


def _init_env():
    """
    Initialize pySMT environment at beginning of each test.
    """
    pysmt.shortcuts.reset_env()
    get_env().enable_infix_notation = True


def test_proof_de_morgan():
    """ Proof de Morgans law.
    """
    _init_env()
    b1, b2 = symbols('b1, b2', BOOL)

    f = b1 & b2
    demorgan = Not(b1) | Not(b2)

    theorem = Iff(Not(f), demorgan)

    assert proof(theorem)


def test_proof_interval_overlap_formula():
    """ Proof interval overlap formula.
    Proof that f2 checks if two intervals overlap given a1<=a2, b1<=b2.
         a1                 a2
          |-----------------|
      b1          b2
       |----------|
    """
    _init_env()

    a1, a2, b1, b2 = symbols('a1, a2, b1, b2', INT)

    # To proove: This should be sufficient to detect if intervals overlap given a1<=a2, b1<=b2.
    f2 = Or(
        (a1 <= b1) & (b1 <= a2),
        (b1 <= a1) & (a1 <= b2),
    )

    """
    f3 = Or(
        (a1 <= b1) & (b1 <= a2),
        (a1 <= b2) & (b2 <= a2),
        (b1 <= a1) & (a1 <= b2)
    )
    
    f4 = Or(
        (a1 <= b1) & (b1 <= a2),
        (a1 <= b2) & (b2 <= a2),
        (b1 <= a1) & (a1 <= b2),
        (b1 <= a2) & (a2 <= b2),
    )
    """

    # Reference.
    # Check for strict ordering of a and b.
    # The intervals don't overlap iff a* < b* OR b* < a*
    no_overlap = Or(
        (a1 < b1) & (a1 < b2) & (a2 < b1) & (a2 < b2),
        (b1 < a1) & (b1 < a2) & (b2 < a1) & (b2 < a2),
    )
    has_overlap = Not(no_overlap)

    # Assume edges are oriented in positive direction.
    assume_edge_direction = (a1 <= a2) & (b1 <= b2)

    # assume_edge_direction => (f2 <-> has_overlap)
    theorem = assume_edge_direction.Implies(Iff(f2, has_overlap))

    assert proof(theorem)
