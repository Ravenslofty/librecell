#
# Copyright (c) 2019-2020 Thomas Kramer.
#
# This file is part of librecell 
# (see https://codeberg.org/tok/librecell).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
from itertools import product
from enum import Enum
import inspect


class Unate(Enum):
    NON_UNATE = 0
    NEGATIVE_UNATE = -1
    POSITIVE_UNATE = 1


def is_unate_in_xi(bool_function, param_name: str) -> Unate:
    """ Test if the boolean function is unate in input `i`.
    A function `f` is positive unate iff: f(..., x_i = 1, ...) >= f(..., x_i = 0, ...)
    A function `f` is negative unate iff: f(..., x_i = 1, ...) <= f(..., x_i = 0, ...)
    `f` is non-unate or binate if it is neither positive unate nor negative unate.
    :param bool_function:
    :param param_name: Name of input variable.
    :return: Unate.NON_UNATE, Unate.NEGATIVE_UNATE or Unate.POSITIVE_UNATE
    """

    # TODO: use sympy to do this more elegantly (satisfiability solver)

    # Get number of inputs to function.

    params = list(inspect.signature(bool_function).parameters)
    assert param_name in params, "Function does not have parameter with name '{}'.".format(param_name)
    variable_params = [p for p in params if p != param_name]
    num_inputs = len(variable_params)

    is_positive_unate = True
    is_negative_unate = True

    fixed_inputs = list(product(*([[0, 1]] * num_inputs)))

    for inp in fixed_inputs:
        inp = list(inp)

        param_values = {n: v for n, v in zip(variable_params, inp)}
        param_values[param_name] = 0

        out0 = bool_function(**param_values)
        param_values[param_name] = 1
        out1 = bool_function(**param_values)

        _is_negative_unate = out1 <= out0
        _is_positive_unate = out1 >= out0

        is_positive_unate = is_positive_unate and _is_positive_unate
        is_negative_unate = is_negative_unate and _is_negative_unate

        is_non_unate = not is_positive_unate and not is_negative_unate
        if is_non_unate:
            break

    is_non_unate = is_positive_unate == is_negative_unate
    if is_non_unate:
        return Unate.NON_UNATE
    elif is_negative_unate:
        return Unate.NEGATIVE_UNATE
    elif is_positive_unate:
        return Unate.POSITIVE_UNATE
    else:
        assert False


def test_is_unate():
    AND = lambda a, b: a and b
    NAND = lambda a, b: not (a and b)
    XOR = lambda a, b: a ^ b

    assert is_unate_in_xi(AND, 'a') == Unate.POSITIVE_UNATE
    assert is_unate_in_xi(AND, 'b') == Unate.POSITIVE_UNATE
    assert is_unate_in_xi(NAND, 'a') == Unate.NEGATIVE_UNATE
    assert is_unate_in_xi(NAND, 'b') == Unate.NEGATIVE_UNATE

    assert is_unate_in_xi(XOR, 'a') == Unate.NON_UNATE
