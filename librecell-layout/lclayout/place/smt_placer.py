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
from z3 import *
from .place import TransistorPlacer
from typing import Iterable
from ..data_types import Transistor, Cell, ChannelType
from itertools import combinations, chain
import logging

logger = logging.getLogger(__name__)


class SMTPlacer(TransistorPlacer):

    def __init__(self):
        pass

    def place(self, transistors: Iterable[Transistor]) -> Cell:
        """
        Place transistors using an SMT solver (Z3).
        :param transistors:
        :return: Placed cell.
        """
        transistors = list(transistors)
        nmos = [t for t in transistors if t.channel_type == ChannelType.NMOS]
        pmos = [t for t in transistors if t.channel_type == ChannelType.PMOS]

        # TODO: expose this as a parameter for multi row cells.
        max_cell_rows = 1

        # Upper bound on cell width.
        max_cell_width = (max(len(nmos), len(pmos)) * 2) // max_cell_rows

        # Optimizer
        opt = Optimize()

        # Wrapper arount opt.add
        def add_assertion(assertion, **kw):
            opt.add(assertion)

        # Create symbols for bounds on transistor positions.
        # Used later to minimize cell width.
        max_x = Int("max_x")
        max_y = Int("max_y")

        # Create symbols for transistor positions.
        transistor_positions = {t: (Int("transistor_{}_x".format(i)), Int("transistor_{}_y".format(i)))
                                for i, t in enumerate(transistors)}

        # Create boolean symbols for transistor flips.
        # Each transistor can be flipped (source/drain swapped).
        transistor_flipped = {t: Bool("transistor_{}_flipped".format(i))
                              for i, t in enumerate(transistors)}

        # Constraint: Positions are bounded.
        # Add bounds on positions.
        for x, y in transistor_positions.values():
            add_assertion(x >= 0)
            add_assertion(y >= 0)

            # Add upper bounds on transistor positions.
            add_assertion(x < max_cell_width)
            add_assertion(y < max_cell_rows * 2)

            add_assertion(max_x >= x)
            add_assertion(max_y >= y)

        # Constraint: Separate rows for NMOS and PMOS
        # Assign rows to NMOS and PMOS
        for t, (x, y) in transistor_positions.items():

            or_constraints = []

            for r in range(max_cell_rows):
                # Place transistor in upper or lower stack?
                # Ordering alternates from row to row such that power stripe can be shared.
                stack = r % 2 if t.channel_type == ChannelType.NMOS else 1 - r % 2

                allowed_y = r * 2 + stack
                or_constraints.append(y == allowed_y)

            on_allowed_row = Or(*or_constraints)
            add_assertion(on_allowed_row)

        # Constraint: Non-overlapping positions
        # No two transistors should have the same position.
        for (x1, y1), (x2, y2) in combinations(transistor_positions.values(), 2):
            same_position = And(
                x1 == x2,
                y1 == y2
            )
            different_positions = Not(same_position)
            add_assertion(different_positions)

        # Constraint: Diffusion sharing
        # If two transistors are placed side-by-side then the abutted sources/drain nets must match.
        for ts in [nmos, pmos]:
            # Loop through all potential (left, right) pairs.
            for a, b in combinations(ts, 2):
                for t_left, t_right in [(a, b), (b, a)]:
                    xl, yl = transistor_positions[t_left]
                    xr, yr = transistor_positions[t_right]

                    # Checks if t_left is left neighbor of t_right.
                    are_neighbors = And(
                        yl == yr,
                        xl + 1 == xr
                    )

                    # Go through all combinations of flipped transistors
                    # and check if they are allowed to be directly abutted if flipped
                    # in a specific way.
                    flip_combinations = [[False, False], [False, True], [True, False], [True, True]]
                    for flip_l, flip_r in flip_combinations:
                        l = t_left.flipped() if flip_l else t_left
                        r = t_right.flipped() if flip_r else t_right

                        if l.right != r.left:
                            # Drain/Source net mismatch.
                            # In case the transistors are flipped that way,
                            # they are not allowed to be direct neighbors.
                            add_assertion(
                                Implies(
                                    And(transistor_flipped[t_left] == flip_l,
                                        transistor_flipped[t_right] == flip_r),
                                    Not(are_neighbors)
                                )
                            )

        # Extract all net names.
        nets = set(chain(*(t.terminals() for t in transistors)))

        # Create net bounds. This will be used to optimize
        # the bounding box perimeter of the nets (for wiring length optimization).
        net_max_x = {net: Int("net_max_x_{}".format(net))
                     for net in nets}

        net_min_x = {net: Int("net_min_x_{}".format(net))
                     for net in nets}

        net_max_y = {net: Int("net_max_y_{}".format(net))
                     for net in nets}

        net_min_y = {net: Int("net_min_y_{}".format(net))
                     for net in nets}

        for t in transistors:
            x, y = transistor_positions[t]

            # TODO: Net positions dependent on transistor terminal.
            #       Now, the net position equals the transistor position.
            #       Make it dependent on the actual terminal (drain, gate, source).
            #       Also depends on transistor flips.
            for net in t.terminals():
                add_assertion(x <= net_max_x[net])
                add_assertion(x >= net_min_x[net])
                add_assertion(y <= net_max_y[net])
                add_assertion(y >= net_min_y[net])

        # Optiimization goals
        # Note: z3 uses lexicographic priorities of objectives by default.
        # Here, the cell width is optimized first.
        # Could be interesting: z3 could also find pareto fronts.

        # Optimization objective 1
        # Minimize cell width.
        opt.minimize(max_x)

        # Optimization objective 2
        # Minimize wiring length (net bounding boxes)
        # TODO: sort criteria by what? Number of terminals?
        for net in nets:
            # TODO: skip VDD/GND nets
            opt.minimize(net_max_x[net] - net_min_x[net])
            opt.minimize(net_max_y[net] - net_min_y[net])

        # TODO: optimization objective for pin nets.

        logger.info("Run SMT optimizer (Z3)")
        sat = opt.check()

        logger.info("SMT result: %s", "Satisfied" if sat else "Unsatisfied")
        if not sat:
            msg = "Placement problem not satisfiable."
            logger.error(msg)
            raise Exception(msg)

        # logger.debug("model = %s", opt.model())

        model = opt.model()
        cell_width = model[max_x].as_long() + 1

        cell = Cell(cell_width)
        rows = [cell.lower, cell.upper]
        for t in transistors:
            x, y = transistor_positions[t]
            x = model[x].as_long()
            y = model[y].as_long()
            flip = is_true(model[transistor_flipped[t]])

            flipped = t.flipped() if flip else t

            rows[y][x] = flipped

        return cell


def test():
    placer = SMTPlacer()
    from itertools import count
    c = count()
    transistors = [Transistor(ChannelType.PMOS, 1, 1, 3, name=next(c)),
                   Transistor(ChannelType.NMOS, 1, 2, 3, name=next(c)),
                   Transistor(ChannelType.PMOS, 1, 1, 3, name=next(c)),
                   Transistor(ChannelType.NMOS, 1, 2, 3, name=next(c)),
                   Transistor(ChannelType.PMOS, 1, 1, 3, name=next(c)),
                   Transistor(ChannelType.NMOS, 1, 2, 3, name=next(c))]
    placer.place(transistors)
