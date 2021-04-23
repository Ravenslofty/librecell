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

"""
Classes for handling piece-wise linear waveforms.
"""

import numpy as np
from scipy import interpolate
from typing import Tuple, Sequence, Iterable, Union
from itertools import chain
import itertools


class PieceWiseLinear:

    def __init__(self, x: Iterable, y: Iterable):
        """
        Define a pice wise linear wave by x,y coordinates.
        :param x: x (time) coordinate
        :param y: y (amplitude) coordinate
        """

        assert all(a <= b for a, b in zip(x, x[1:])), "x axis must be sorted."

        self.x = np.array(x)
        self.y = np.array(y)

    def concatenate(self, other):
        # TODO: untested
        return PieceWiseLinear(
            np.concatenate([self.x, other.x + self.x[-1]]),
            np.concatenate([self.y, other.y])
        )

    def split_at(self, x: Union[int, float]) -> Tuple:
        # TODO: untested
        index = np.max(np.argwhere(self.x < x) + 1, initial=0)
        self.add_sampling_point(x)

        return PieceWiseLinear(self.x[:index + 1], self.y[:index + 1]), \
               PieceWiseLinear(self.x[index:], self.y[:index])

    def interpolated(self):
        """
        Represent the wave form as a function of x.
        Interpolation: Linear
        Extrapolation: Using the closest defined value.
        :return: Function of x
        """
        return interpolate.interp1d(self.x, self.y,
                                    bounds_error=False,
                                    fill_value=(self.y[0], self.y[-1]))

    def add_sampling_point(self, x: Union[int, float]):
        """
        Inserts a redundant (x,y) sample point into the wave. y is determined by interpolation
        or extrapolation. The waveform itself is not changed.
        This is used if the raw x,y arrays are required to be defined at certain intervals.
        :param x: x offset
        :return:
        """

        if x in self.x:
            return

        y = self(x)
        index = np.max(np.argwhere(self.x < x) + 1, initial=0)

        self.x = np.insert(self.x, index, x)
        self.y = np.insert(self.y, index, y)

    def shifted(self, delta_t):
        """
        Get the same wave shifted by `delta_t`
        :param delta_t: Time shift.
        :return: f(t + delta_t)
        """
        return PieceWiseLinear(
            x=self.x + delta_t,
            y=self.y.copy()
        )

    def __add__(self, other):
        """
        Add two functions.
        Input functions are assumed to be 0 ouside of their definition range.
        :param other:
        :return: PieceWiseLinearWave
        """

        if isinstance(other, PieceWiseLinear):

            f1 = self.interpolated()
            f2 = other.interpolated()

            # Concatenate, remove duplicates and sort by time.
            x = set(chain(self.x, other.x))
            x = np.array(sorted(set(x)))

            y = f1(x) + f2(x)

            return PieceWiseLinear(x, y)
        else:
            return PieceWiseLinear(self.x.copy(), self.y + other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        """
        Multiplication by a scalar.
        :param other: Scalar
        :return: PieceWiseLinearWave
        """
        assert isinstance(other, float) or isinstance(other, int)
        return PieceWiseLinear(self.x.copy(), self.y * other)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        negy = -self.y
        return PieceWiseLinear(self.x.copy(), negy)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __call__(self, x):
        return self.interpolated()(x)

    def to_spice_pwl_string(self):
        """
        Format the signal as it is needed for describing a PWL source in SPICE.
        The format is like: "T1 V1 T2 V2 ..." Where T is a time in seconds, V a voltage in volts.
        :return:
        """
        pwl_string = ' '.join((
            f'{time:0.20e}s {voltage:0.20e}V'
            for time, voltage in zip(self.x, self.y)
        ))
        return pwl_string


class StepWave(PieceWiseLinear):
    def __init__(self, start_time,
                 polarity: bool = True,
                 transition_time=0.0,
                 rise_threshold=0.5,
                 fall_threshold=0.5):
        assert 0 <= rise_threshold <= 1
        assert 0 <= fall_threshold <= 1
        assert transition_time >= 0

        transition_threshold = rise_threshold if polarity else 1 - fall_threshold

        start = start_time - transition_threshold * transition_time
        end = start_time + ((1 - transition_threshold) * transition_time)

        if polarity:
            y = [0, 1]
        else:
            y = [1, 0]

        super().__init__(
            x=[start, end],
            y=y
        )


def test_step_wave():
    from math import isclose
    rising_edge = StepWave(start_time=0, polarity=True, transition_time=1,
                           rise_threshold=0.8, fall_threshold=0.2)

    assert isclose(rising_edge(-0.8), 0)
    assert rising_edge(-0.79999) > 0
    assert isclose(rising_edge(0), 0.8)
    assert rising_edge(0.19999) < 1
    assert isclose(rising_edge(0.2), 1)

    falling_edge = StepWave(start_time=0, polarity=False, transition_time=1,
                            rise_threshold=0.8, fall_threshold=0.2)

    assert isclose(falling_edge(-0.8), 1)
    assert falling_edge(-0.79999) < 1
    assert isclose(falling_edge(0), 0.2)
    assert falling_edge(0.19999) > 0
    assert isclose(falling_edge(0.2), 0)


class PulseWave(PieceWiseLinear):

    # TODO: also use slew thresholds. Maybe pass TripPoint object to constructor.
    def __init__(self, start_time: float,
                 duration: float,
                 polarity: bool = True,
                 rise_time=0.0,
                 fall_time=0.0,
                 rise_threshold=0.5,
                 fall_threshold=0.5):
        """
        Create a single pulse with given start time and duration.
        The duration is measured from the crossing of rise_threshold to fall_threshold.

        :param start_time:
        :param duration:
        :param polarity: True: create a HIGH pulse, False: create a LOW pulse.
        :param rise_time:
        :param fall_time:
        :param rise_threshold:
        :param fall_threshold:
        """
        assert 0 <= rise_threshold <= 1
        assert 0 <= fall_threshold <= 1
        assert duration >= 0

        if polarity:
            start1 = start_time - rise_threshold * rise_time
            end1 = start_time + ((1 - rise_threshold) * rise_time)
            start2 = start_time + duration - (1 - fall_threshold) * fall_time
            end2 = start_time + duration + fall_threshold * fall_time
        else:
            start1 = start_time - (1 - fall_threshold) * fall_time
            end1 = start_time + fall_threshold * fall_time
            start2 = start_time + duration - rise_threshold * rise_time
            end2 = start_time + duration + (1 - rise_threshold) * rise_time

        assert start1 <= end1
        assert start2 <= end2

        assert start1 <= end1 <= start2 <= end2

        if polarity:
            y = [0, 1, 1, 0]
        else:
            y = [1, 0, 0, 1]

        super().__init__(
            x=[start1, end1, start2, end2],
            y=y
        )


def test_piece_wise_linear():
    p1 = PieceWiseLinear([0, 1, 2, 3],
                         [0, 1, 1, 0])

    p2 = PieceWiseLinear([0, 1, 2, 3, 4],
                         [0, 1, 1, 1, 1])

    p3 = p1 + p2

    p4 = p2 - p1

    p5 = p1 * 1.23

    for x in [0, 0.5, 1, 2, 2.5, 3, 4, 5]:
        assert p3(x) == p1(x) + p2(x)
        assert p4(x) == p2(x) - p1(x)
        assert p5(x) == p1(x) * 1.23


def test_pulse_wave():
    pulse = PulseWave(start_time=10, duration=10, rise_time=2, fall_time=2)
    pulse_inv = PulseWave(start_time=10, duration=10, rise_time=2, fall_time=2, polarity=False)

    f = pulse.interpolated()

    x = [-1, 9, 10, 11, 15, 19, 20, 21, 100]
    expected = np.array([0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])
    expected_inv = 1 - expected

    actual = f(x)

    assert all(expected == actual)
    assert all(expected_inv == pulse_inv.interpolated()(x))


def test_pulse_wave_negative():
    from math import isclose
    p = PulseWave(start_time=0,
                  duration=10,
                  rise_time=1,
                  fall_time=2,
                  polarity=False,
                  rise_threshold=0.8,
                  fall_threshold=0.2)

    assert isclose(p(-2), 1)
    assert isclose(p(-1.6), 1)
    assert p(-0.7999) < 1
    assert isclose(p(0), 0.2)
    assert isclose(p(5), 0)
    assert isclose(p(10), 0.8)
    assert p(10.1999) < 1
    assert isclose(p(10.2), 1)


def test_pulse_wave_positive():
    from math import isclose
    p = PulseWave(start_time=0,
                  duration=10,
                  rise_time=2,
                  fall_time=1,
                  polarity=True,
                  rise_threshold=0.8,
                  fall_threshold=0.2)

    assert isclose(p(-2), 0)
    assert isclose(p(-1.6), 0)
    assert p(-0.7999) > 0
    assert isclose(p(0), 0.8)
    assert isclose(p(5), 1)
    assert isclose(p(10), 0.2)
    assert p(10.1999) > 0
    assert isclose(p(10.2), 0)


def bitsequence_to_piece_wise_linear(bits: Sequence[int],
                                     bit_duration: float,
                                     rise_time: float = 0,
                                     fall_time: float = 0,
                                     rise_threshold: float = 0.0,
                                     fall_threshold: float = 1.0,
                                     start_time: float = 0) -> PieceWiseLinear:
    """ Generate a piece wise linear waveform from a sequence of bits.

    :param bits: Sequence of bits {0,1}*.
    :param bit_duration: Duration of one bit.
    :param rise_time: Required time to change from 0 to 1.
    :param fall_time: Required time to change from 1 to 0.
    :return: A tuple of two `np.ndarray`s (time, voltage).
    """
    pulse_durations = [(bit, len(list(g))) for bit, g in itertools.groupby(bits)]

    durations = []
    start_times = []
    t = 0
    for bit, l in pulse_durations:
        if bit:
            assert l > 0
            durations.append(l * bit_duration)
            start_times.append(t * bit_duration + start_time)
        t += l

    pulses = [PulseWave(start_time=start,
                        duration=duration,
                        rise_time=rise_time,
                        fall_time=fall_time,
                        rise_threshold=rise_threshold,
                        fall_threshold=fall_threshold)
              for start, duration in zip(start_times, durations)]

    wave = sum(pulses)
    wave.add_sampling_point(0)
    # wave.add_sampling_point(10)
    return wave


def bitsequence_to_piece_wise_linear_old(bits: Sequence[int],
                                         bit_duration: float,
                                         rise_time: float = 0,
                                         fall_time: float = 0,
                                         start_time: float = 0) -> PieceWiseLinear:
    """ Generate a piece wise linear waveform from a sequence of bits.

    :param bits: Sequence of bits {0,1}*.
    :param bit_duration: Duration of one bit.
    :param rise_time: Required time to change from 0 to 1.
    :param fall_time: Required time to change from 1 to 0.
    :return: A tuple of two `np.ndarray`s (time, voltage).
    """
    previous = 0
    times = [min(start_time, 0)]  # Make sure voltage is always defined starting from time 0.
    voltages = [previous]
    for i, b in enumerate(bits):
        if b != previous:
            t0 = i * bit_duration + start_time

            times.append(t0)
            voltages.append(previous)

            t1 = t0 + (rise_time if b else fall_time)
            times.append(t1)
            voltages.append(b)

            previous = b

    # # Hold the signal at the last value for a second.
    # times.append(start_time + len(bits) * bit_duration + 1)
    # voltages.append(previous)

    return PieceWiseLinear(np.array(times), np.array(voltages))


def test_bitsequence_to_piece_wise_linear():
    bits = [0, 1, 1, 0]
    wave = bitsequence_to_piece_wise_linear(bits, 10, rise_time=1, fall_time=2,
                                            rise_threshold=0, fall_threshold=1)

    assert all(wave.x == np.array([0, 10, 11, 30, 32]))
    assert all(wave.y == np.array([0, 0, 1, 1, 0]))
