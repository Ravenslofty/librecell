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
from itertools import product
import numpy as np
from typing import Sequence, Tuple, Iterator


class Grid:

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int], step: Tuple[int, int]):
        self.start = start
        self.end = end
        self.step = step

        self._dimension = len(start)

        assert all([len(v) == self._dimension for v in (start, end, step)]), \
            "start, end and step must have the same dimensions."

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        """ Get iterator over grid points.
        :return:
        """

        return product(*(range(a, b, c)
                         for a, b, c in zip(self.start, self.end, self.step)
                         ))

    def meshgrid(self):
        """ Return the grid points as a numpy meshgrid.
        :return:
        """
        return np.meshgrid(*(np.arange(a, b, c)
                             for a, b, c in zip(self.start, self.end, self.step)
                             ))

    def _is_on_grid_axis(self, point: Sequence[int], dimension: int) -> bool:
        """ Check if `point` is aligned with grid regarding one dimension.
        :param point:
        :param dimension:
        :return:
        """
        x = point[dimension]
        start = self.start[dimension]
        step = self.step[dimension]
        end = self.end[dimension]

        return (x - start) % step == 0 and start <= x < end

    def is_on_grid(self, point) -> bool:
        return all((self._is_on_grid_axis(point, d) for d in range(self._dimension)))

    def _grid_floor(self, x: int, dimension: int) -> int:
        """ Round a coordinate down to next grid coordinate.
        """
        start = self.start[dimension]
        step = self.step[dimension]
        x = int(x)
        x = (x - start) // step * step + start

        return x

    def _grid_ceil(self, x: int, dimension: int) -> int:
        """ Round a coordinate up to next grid coordinate.
        """
        return self._grid_floor(x + self.step[dimension] - 1, dimension)

    def _grid_round(self, x: int, dimension: int) -> int:
        return self._grid_floor(x + self.step[dimension] // 2, dimension)


class Grid2D(Grid):

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int], step: Tuple[int, int]):
        super().__init__(start, end, step)
        assert self._dimension == 2, "Dimension must be 2."

    def grid_floor_x(self, point):
        x, y = point
        return self._grid_floor(x, 0), y

    def grid_floor_y(self, point):
        x, y = point
        return x, self._grid_floor(y, 1)

    def grid_floor_xy(self, point):
        x, y = point
        return self._grid_floor(x, 0), self._grid_floor(y, 1)

    def grid_ceil_x(self, point):
        x, y = point
        return self._grid_ceil(x, 0), y

    def grid_ceil_y(self, point):
        x, y = point
        return x, self._grid_ceil(y, 1)

    def grid_ceil_xy(self, point):
        x, y = point
        return self._grid_ceil(x, 0), self._grid_ceil(y, 1)

    def grid_round_x(self, point):
        x, y = point
        return self._grid_round(x, 0), y

    def grid_round_y(self, point):
        x, y = point
        return x, self._grid_round(y, 1)

    def grid_round_xy(self, point):
        x, y = point
        return self._grid_round(x, 0), self._grid_round(y, 1)

    def neigborhood(self, point: Tuple[int, int], max_distance: int, norm_ord=2):
        """ Get all points on grid that are at most `max_distance` away from `point`.
        :param point: Reference point.
        :param max_distance:
        :param norm_ord: The order of the norm to be used. Default = 2 (euclidian norm)
        :return:
        """
        offset = self.grid_floor_xy(point)
        start = self.grid_floor_xy((offset[0] - max_distance, offset[1] - max_distance))
        end = self.grid_ceil_xy(
            (offset[0] + max_distance + self.step[0] + 1, offset[1] + max_distance + self.step[1] + 1))

        kernel = Grid2D(start, end, self.step)

        mesh = kernel.meshgrid()
        xx, yy = mesh
        x, y = point
        diff = xx - x, yy - y
        dists = np.linalg.norm(np.array(diff), ord=norm_ord, axis=0)

        neighbor_mask = dists <= max_distance

        neighbor_points = zip(*(x[neighbor_mask].flat for x in mesh))
        return [p for p in neighbor_points if self.is_on_grid(p)]


def test_grid2d():
    g = Grid2D((0, 0), (10, 20), (1, 2))

    assert g.is_on_grid((0, 0))
    assert not g.is_on_grid((1, 1))
    assert g.is_on_grid((1, 2))

    assert g.grid_floor_xy((1, 1)) == (1, 0)
    assert g.grid_ceil_xy((1, 1)) == (1, 2)


def test_grid_neighbors():
    for i in range(0, 1000):
        start = (np.random.randint(-10, 10), np.random.randint(-10, 10))
        end = (np.random.randint(10, 20), np.random.randint(10, 20))
        step = (np.random.randint(1, 4), np.random.randint(1, 4))
        g = Grid2D(start, end, step)
        point = np.random.random(2) * 10
        max_dist = np.random.random(1) * 10
        norm_ord = np.random.randint(1, 3)
        neighbors_expected = [p for p in g if np.linalg.norm(np.array(p) - np.array(point), ord=norm_ord) <= max_dist]
        neighbors_actual = g.neigborhood(point, max_dist, norm_ord=norm_ord)

        # print(set(neighbors_actual).symmetric_difference(set(neighbors_expected)))

        assert set(neighbors_actual) == set(neighbors_expected)
