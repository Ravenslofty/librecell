#
# Copyright 2021 Dan Ravensloft.
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
from .place import TransistorPlacer

from ..data_types import *

from itertools import product
from math import sqrt
from random import Random
from typing import Iterable, List, Tuple

import logging

logger = logging.getLogger(__name__)

def _assemble_cell(lower_row: List[Transistor], upper_row: List[Transistor]) -> Cell:
    """ Build a Cell object from a nmos and pmos row.
    :param lower_row:
    :param upper_row:
    :return:
    """
    width = max(len(lower_row), len(upper_row))
    cell = Cell(width)
    for i, t in enumerate(upper_row):
        cell.upper[i] = t

    for i, t in enumerate(lower_row):
        cell.lower[i] = t
    return cell


def _validate(lower_row: List[Transistor], upper_row: List[Transistor]) -> Tuple[bool, str]:
    for (i, lower) in enumerate(lower_row):
        if lower is not None and lower.channel_type == ChannelType.PMOS:
            logger.error((lower_row, upper_row))
            return False, "PMOS on lower row location {}".format(i)

        if lower is not None and i + 1 < len(lower_row) and lower_row[i + 1] is not None and lower.drain_net != lower_row[i + 1].source_net:
            logger.error((lower_row, upper_row))
            return False, "Missing diffusion gap on lower row location {}".format(i)

    for (i, upper) in enumerate(upper_row):
        if upper is not None and upper.channel_type == ChannelType.NMOS:
            logger.error((lower_row, upper_row))
            return False, "NMOS on upper row location {}".format(i)

        if upper is not None and i + 1 < len(upper_row) and upper_row[i + 1] is not None and upper.drain_net != upper_row[i + 1].source_net:
            logger.error((lower_row, upper_row))
            return False, "Missing diffusion gap on upper row location {}".format(i)

    return True, ""


def _legalise(lower_row: List[Transistor], upper_row: List[Transistor], verbose=False) -> Tuple[List[Transistor], List[Transistor]]:
    def _traverse(lower_row: List[Transistor], upper_row: List[Transistor]) -> Tuple[List[Transistor], List[Transistor]]:
        """Legalise transistor rows.
        :param lower_row:
        :param upper_row:
        :return:
        """

        if verbose:
            print("---")
            print(upper_row)
            print(lower_row)
            print("---")

        if len(lower_row) == 0 and len(upper_row) == 0:
            return lower_row, upper_row

        if len(lower_row) < len(upper_row):
            if verbose:
                print("*** Padding lower row of length {} to match upper row length {}".format(len(lower_row), len(upper_row)))
            return _traverse(lower_row + [None], upper_row)

        if len(lower_row) > len(upper_row):
            if verbose:
                print("*** Padding upper row of length {} to match lower row length {}".format(len(upper_row), len(lower_row)))
            return _traverse(lower_row, upper_row + [None])

        # Constrain transistors to the correct row.
        if len(lower_row) >= 1 and lower_row[0] is not None and lower_row[0].channel_type == ChannelType.PMOS:
            if verbose:
                print("*** PMOS on lower row; moving to upper row")
            return _traverse([None] + lower_row[1:], upper_row[0:1] + lower_row[0:1] + upper_row[1:])

        if len(upper_row) >= 1 and upper_row[0] is not None and upper_row[0].channel_type == ChannelType.NMOS:
            if verbose:
                print("*** NMOS on upper row; moving to lower row")
            return _traverse(lower_row[0:1] + upper_row[0:1] + lower_row[1:], [None] + upper_row[1:])

        # Insert diffusion gaps for non-matching nets.
        if len(lower_row) >= 2 and lower_row[0] is not None and lower_row[1] is not None and lower_row[0].drain_net != lower_row[1].source_net:
            if verbose:
                print("*** Inserting lower row diffusion gap")
            return _traverse(lower_row[0:1] + [None] + lower_row[1:], upper_row)

        if len(upper_row) >= 2 and upper_row[0] is not None and upper_row[1] is not None and upper_row[0].drain_net != upper_row[1].source_net:
            if verbose:
                print("*** Inserting upper row diffusion gap")
            return _traverse(lower_row, upper_row[0:1] + [None] + upper_row[1:])

        lower_tail, upper_tail = _traverse(lower_row[1:], upper_row[1:])
        return (lower_row[0:1] + lower_tail, upper_row[0:1] + upper_tail)

    start_lower, start_upper = lower_row, upper_row

    lower_row = [x for x in lower_row if x is not None]
    upper_row = [x for x in upper_row if x is not None]

    lower_row, upper_row = _traverse(lower_row, upper_row)

    while lower_row[-1] is None and upper_row[-1] is None:
        lower_row.pop()
        upper_row.pop()

    ok, msg = _validate(lower_row, upper_row)
    if not ok:
        logger.error((start_lower, start_upper))
        assert ok, msg

    return lower_row, upper_row


def _evaluate(lower_row: List[Transistor], upper_row: List[Transistor]) -> float:
    # First, ensure the rows are legal.
    lower_row, upper_row = _legalise(lower_row, upper_row)

    assert len(lower_row) >= 1, "Placement result has no transistors"
    assert len(upper_row) >= 1, "Placement result has no transistors"

    # Count the distance between nets
    nets = {}
    for x, node in enumerate(lower_row):
        if node is not None:
            coord = (x, 0)
            if node.source_net not in nets:
                nets[node.source_net] = [coord]
            else:
                nets[node.source_net].append(coord)

            if node.gate_net not in nets:
                nets[node.gate_net] = [coord]
            else:
                nets[node.gate_net].append(coord)

            if node.drain_net not in nets:
                nets[node.drain_net] = [coord]
            else:
                nets[node.drain_net].append(coord)

    for node in upper_row:
        if node is not None:
            coord = (x, 1)
            if node.source_net not in nets:
                nets[node.source_net] = [coord]
            else:
                nets[node.source_net].append(coord)

            if node.gate_net not in nets:
                nets[node.gate_net] = [coord]
            else:
                nets[node.gate_net].append(coord)

            if node.drain_net not in nets:
                nets[node.drain_net] = [coord]
            else:
                nets[node.drain_net].append(coord)

    distance = 0
    for net in nets.values():
        for (a, b) in product(net, net):
            ax, ay = a
            bx, by = b
            distance += abs(ax - bx) + abs(ay - by)

    return distance


class RandomPlacer(TransistorPlacer):
    def __init__(self):
        # For reproducibility, seed the generator with a fixed constant.
        self.rand = Random(1)

    def place(self, transistors: Iterable[Transistor]) -> Cell:
        """Place transistors randomly.
        :param transistors:
        :return:
        """
        # First, shuffle the order of the transistors.
        self.rand.shuffle(transistors)

        # For additional randomness, shuffle transistor sources and drain.
        for transistor in transistors:
            if self.rand.choice([False, True]):
                transistor.source_net, transistor.drain_net = transistor.drain_net, transistor.source_net

        # Legalise placement.
        l = len(transistors)
        lower_row, upper_row = _legalise(transistors[:l//2], transistors[l//2:])
        ok, msg = _validate(lower_row, upper_row)
        assert ok, msg

        return _assemble_cell(lower_row, upper_row)


def _neighbour(rand: Random, lower_row: List[Transistor], upper_row: List[Transistor]) -> Tuple[List[Transistor], List[Transistor]]:
    # Filter out gaps from the rows; _legalise will re-insert them where needed.
    lower_row = [x for x in lower_row if x is not None]
    upper_row = [x for x in upper_row if x is not None]

    # Should we swap a transistor's source and drain, or should we swap the positions of two transistors?
    pinswap = rand.choice([False, True])

    # Upper or lower row?
    upper = rand.choice([False, True])

    if pinswap:
        if upper:
            node = rand.choice(upper_row)
            node.source_net, node.drain_net = node.drain_net, node.source_net
        else:
            node = rand.choice(lower_row)
            node.source_net, node.drain_net = node.drain_net, node.source_net
    else:
        if upper:
            a, b = rand.randrange(0, len(upper_row)), rand.randrange(0, len(upper_row))

            while True:
                if a == b:
                    a, b = rand.randrange(0, len(upper_row)), rand.randrange(0, len(upper_row))
                else:
                    break

            upper_row[a], upper_row[b] = upper_row[b], upper_row[a]
        else:
            a, b = rand.randrange(0, len(lower_row)), rand.randrange(0, len(lower_row))

            while True:
                if a == b:
                    a, b = rand.randrange(0, len(lower_row)), rand.randrange(0, len(lower_row))
                else:
                    break

            lower_row[a], lower_row[b] = lower_row[b], lower_row[a]

    return lower_row, upper_row


class HillClimbPlacer(TransistorPlacer):
    def place(self, transistors: Iterable[Transistor]) -> Cell:
        NO_IMPROVEMENT = 100

        # Generate a random starting cell.
        placer = RandomPlacer()
        total_best_cell = placer.place(transistors)
        total_best_score = _evaluate(total_best_cell.lower, total_best_cell.upper)
        total_sideways = NO_IMPROVEMENT

        for metaiteration in range(1000):
            if total_sideways == 0:
                break

            best_cell = placer.place(transistors)
            best_score = _evaluate(best_cell.lower, best_cell.upper)

            sideways = NO_IMPROVEMENT

            for iteration in range(1000):
                # If we have not found any improvement after a while searching, give up.
                if sideways == 0:
                    # logger.debug("Iteration %d - no improvement after %d iterations" % (iteration, NO_IMPROVEMENT))
                    break

                lower_row, upper_row = _neighbour(placer.rand, best_cell.lower, best_cell.upper)
                lower_row, upper_row = _legalise(lower_row, upper_row)
                score = _evaluate(lower_row, upper_row)

                if score < best_score:
                    best_score = score
                    best_cell = _assemble_cell(lower_row, upper_row)
                    # logger.debug("Iteration %d - improved efficiency: %.02f%%" % (iteration, 100.0 * best_score))
                    sideways = NO_IMPROVEMENT
                elif abs(score - best_score) < 0.0001:
                    best_score = score
                    best_cell = _assemble_cell(lower_row, upper_row)
                    # logger.debug("Iteration %d - accepting roughly equal efficiency: %.02f%%" % (iteration, 100.0 * best_score))
                    sideways -= 1
                else:
                    sideways -= 1

            logger.debug("Meta-iteration %d finish score: %.02f best so far: %.02f" % (metaiteration, best_score, total_best_score))

            if best_score < total_best_score:
                total_best_score = best_score
                total_best_cell = best_cell
                total_sideways = 10
            else:
                total_sideways -= 1

        logger.debug("Finish score: %.02f" % (total_best_score))

        return total_best_cell


class ThresholdAcceptancePlacer(TransistorPlacer):
    def place(self, transistors: Iterable[Transistor]) -> Cell:
        LIST_SIZE = 5000 * len(transistors) // 22
        STEPS_PER_ITER = 20000 * len(transistors) // 22

        placer = RandomPlacer()
        total_best_cell = placer.place(transistors)
        total_best_score = _evaluate(total_best_cell.lower, total_best_cell.upper)

        optimal = len(transistors) # If we achieve a 100% packing, call it a day.

        logger.info("Starting best score: {}; cell dimensions: {}x2".format(total_best_score, len(total_best_cell.lower)))

        # Explore the neighbourhood to find a roughly appropriate series of thresholds.
        logger.info("Pre-solve")
        threshold = []
        iteration = 0
        while iteration < LIST_SIZE:
            lower_row, upper_row = _neighbour(placer.rand, total_best_cell.lower, total_best_cell.upper)
            lower_row, upper_row = _legalise(lower_row, upper_row)
            score = _evaluate(lower_row, upper_row)
            if score < total_best_score:
                total_best_score = score
                total_best_cell = _assemble_cell(lower_row, upper_row)
                logger.info("New best score: {}; cell dimensions: {}x2".format(total_best_score, len(lower_row)))
                if len(lower_row) + len(upper_row) == optimal:
                    logger.debug("100% packing achieved!")
                    logger.debug("Finish score: {}".format(total_best_score))
                    return total_best_cell
            elif score > total_best_score:
                threshold.append((score - total_best_score) / total_best_score)
                iteration += 1

        logger.info("Solve")
        best_cell = placer.place(transistors)
        best_score = _evaluate(total_best_cell.lower, total_best_cell.upper)
        iteration = 0
        while iteration < STEPS_PER_ITER:
            if iteration % 1000 == 0 and iteration != 0:
                logger.info("Boredom: {}/{}".format(iteration, STEPS_PER_ITER))

            lower_row, upper_row = _neighbour(placer.rand, best_cell.lower, best_cell.upper)
            lower_row, upper_row = _legalise(lower_row, upper_row)
            score = _evaluate(lower_row, upper_row)
            iteration += 1
            if score <= best_score:
                best_score = score
                best_cell = _assemble_cell(lower_row, upper_row)
                if score <= total_best_score:
                    total_best_score = score
                    total_best_cell = best_cell
                    logger.info("New best score: {}; cell dimensions: {}x2".format(total_best_score, len(lower_row)))
                    if len(lower_row) + len(upper_row) == optimal:
                        logger.debug("100% packing achieved!")
                        logger.debug("Finish score: {}".format(total_best_score))
                        return total_best_cell
                iteration = 0
            else:
                max_pos = 0
                max_val = 0
                for i, x in enumerate(threshold):
                    if x > max_val:
                        max_pos = i
                        max_val = x
                change = (score - best_score) / best_score
                if change < max_val:
                    threshold[max_pos] = change
                    best_score = score
                    best_cell = _assemble_cell(lower_row, upper_row)
                    iteration = 0

        logger.debug("Finish score: {}".format(total_best_score))

        return total_best_cell


if __name__ == "__main__":
    lower_row, upper_row = ([Transistor(ChannelType.PMOS, 'VDD', '5', '4'), Transistor(ChannelType.PMOS, '6', '1', '7'), Transistor(ChannelType.NMOS, '9', '1', '3'), Transistor(ChannelType.PMOS, '5', '3', 'VDD'), Transistor(ChannelType.NMOS, 'GND', '5', '11'), Transistor(ChannelType.NMOS, '7', '1', '12'), Transistor(ChannelType.PMOS, '3', 'CLK', '2'), Transistor(ChannelType.PMOS, 'VDD', '7', 'Q'), Transistor(ChannelType.NMOS, 'GND', 'D', '9'), Transistor(ChannelType.PMOS, '6', '5', 'VDD'), Transistor(ChannelType.NMOS, 'GND', 'Q', '12')], [Transistor(ChannelType.PMOS, '7', 'CLK', '8'), Transistor(ChannelType.PMOS, '4', '1', '3'), Transistor(ChannelType.NMOS, '10', 'CLK', '3'), Transistor(ChannelType.PMOS, '1', 'CLK', 'VDD'), Transistor(ChannelType.NMOS, 'GND', 'CLK', '1'), Transistor(ChannelType.NMOS, '10', '5', 'GND'), Transistor(ChannelType.PMOS, '2', 'D', 'VDD'), Transistor(ChannelType.PMOS, '8', 'Q', 'VDD'), Transistor(ChannelType.NMOS, '5', '3', 'GND'), Transistor(ChannelType.NMOS, '11', 'CLK', '7'), Transistor(ChannelType.NMOS, 'Q', '7', 'GND')])
    lower_row, upper_row = _legalise(lower_row, upper_row, verbose=False)
    ok, msg = _validate(lower_row, upper_row)
    assert ok, msg
