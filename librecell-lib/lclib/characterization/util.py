##
## Copyright (c) 2019 Thomas Kramer.
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
import numpy as np
from scipy import interpolate, optimize
from enum import Enum
from collections import namedtuple
from liberty.types import Group
import logging

logger = logging.getLogger(__name__)


class TimingCorner(Enum):
    WORST = 1
    TYPICAL = 2
    BEST = 3


TripPoints = namedtuple("TripPoints", [
    "input_threshold_rise",
    "input_threshold_fall",
    "output_threshold_rise",
    "output_threshold_fall",

    "slew_lower_threshold_rise",
    "slew_upper_threshold_rise",
    "slew_lower_threshold_fall",
    "slew_upper_threshold_fall"
])


#
# # TODO: Add type hints for Python 3.6.
# class TripPoints(NamedTuple):
#     input_threshold_rise = 0.5
#     input_threshold_fall = 0.5
#     output_threshold_rise = 0.5
#     output_threshold_fall = 0.5
#
#     slew_lower_threshold_rise = 0.2
#     slew_upper_threshold_rise = 0.8
#     slew_lower_threshold_fall = 0.2
#     slew_upper_threshold_fall = 0.8


def is_rising_edge(voltage: np.ndarray, threshold: float = 0.5) -> bool:
    """ Check if the signal rises by comparing first and last value.
    :param voltage: Signal.
    :param threshold: Decision threshold for HIGH/LOW value.
    :return: True iff the signal rises.
    """
    return voltage[0] < threshold < voltage[-1]


def is_falling_edge(voltage: np.ndarray, threshold: float = 0.5) -> bool:
    """ Check if the signal falls by comparing first and last value.
    :param voltage: Signal.
    :param threshold: Decision threshold for HIGH/LOW value.
    :return: True iff the signal falls.
    """
    return voltage[0] > threshold > voltage[-1]


def transition_time(voltage: np.ndarray, time: np.ndarray,
                    threshold: float, n: int = -1,
                    assert_one_crossing: bool = False) -> float:
    """ Find time of the n-th event when the signal crosses the threshold.
    :param voltage: np.ndarray holding voltage values.
    :param time: np.ndarray holding time values.
    :param threshold:
    :param n: Selects the event if there are multiple. 0: first event, -1: last event.
    :param assert_one_crossing: If set, then assert that the signal crosses the threshold exactly once.
    :return: Time when the signal crosses the threshold for the n-th time.
    """

    y_shifted = voltage - threshold

    # Find zero-crossings.
    # 0: no zero crossing here
    # 1: crossing from negative to positive
    # -1: crossing from positive to negative
    transitions = np.sign(np.diff(np.sign(y_shifted)))
    index = np.arange(len(transitions))
    # Get indices of crossings.
    transition_indices = index[transitions != 0]

    transition_idx = transition_indices[n]

    if assert_one_crossing:
        # Count number of zero-crossings. There should be exactly one.
        assert np.sum(transitions != 0) > 0, "Signal does not cross threshold."
        assert np.sum(transitions != 0) == 1, "Signal crosses threshold multiple times."

    is_rising = transitions[transition_idx] == 1

    if not is_rising:
        # Normalize to rising edge.
        y_shifted = -y_shifted

    # Estimate where `y` crosses `threshold`.
    estimate = time[transition_idx]

    # Interpolate the samples find a more accurate time of threshold crossing.
    f_interp = interpolate.interp1d(time, y_shifted)

    threshold_cross_arg = optimize.bisect(f_interp, time[transition_idx - 1], time[transition_idx + 1])
    return threshold_cross_arg


def get_slew_time(time: np.ndarray, voltage: np.ndarray,
                  trip_points: TripPoints) -> float:
    """
    Calculate the slew time of `voltage` signal.
    :param time: Time axis.
    :param voltage:
    :param trip_points:
    :return: Slew time in same units as `time`.
    """

    if is_falling_edge(voltage, trip_points.slew_lower_threshold_fall):
        threshold1 = trip_points.slew_upper_threshold_fall
        threshold2 = trip_points.slew_lower_threshold_fall
    elif is_rising_edge(voltage, trip_points.slew_upper_threshold_rise):
        threshold1 = trip_points.slew_lower_threshold_rise
        threshold2 = trip_points.slew_upper_threshold_rise
    else:
        assert False, "Signal has neither a rising edge nor a falling edge."

    slew = transition_time(voltage, time, threshold2) - transition_time(voltage, time, threshold1)

    assert slew >= 0, "Slew can't be negative."

    return slew


def get_input_to_output_delay(time: np.ndarray, input_signal: np.ndarray, output_signal: np.ndarray,
                              trip_points: TripPoints) -> float:
    """Calculate delay from the moment the input signal crosses `trip_points.input_threshold_{rise,fall}`
    to the moment the output signal crosses `trip_points.output_threshold_{rise,fall}`.

    Rise/fall thresholds are selected automatically depending on start and end values of the signals.

    :param time: Time axis.
    :param input_signal: Voltage of input signal (probably normalized to [0,1]).
    :param output_signal: Voltage of output signal (probably normalized to [0,1]).
    :param trip_points: TripPoints object holding threshold values.
    :return: Delay in same units as `time`.
    """

    if is_falling_edge(input_signal, trip_points.input_threshold_fall):
        i_threshold = trip_points.input_threshold_fall
    elif is_rising_edge(input_signal, trip_points.input_threshold_rise):
        i_threshold = trip_points.input_threshold_rise
    else:
        assert False, "Input signal has neither a rising edge nor a falling edge."

    if is_falling_edge(output_signal, trip_points.output_threshold_fall):
        o_threshold = trip_points.output_threshold_fall
    elif is_rising_edge(output_signal, trip_points.output_threshold_rise):
        o_threshold = trip_points.output_threshold_rise
    else:
        assert False, "Output signal has neither a rising edge nor a falling edge."

    delay = transition_time(output_signal, time, o_threshold) - transition_time(input_signal, time, i_threshold)
    assert isinstance(delay, float)
    return delay


def read_trip_points_from_liberty(library_group: Group) -> TripPoints:
    """
    Read trip points from a liberty library Group object.
    :param library_group:
    :return:
    """
    assert library_group.group_name == 'library', "Expected a `library` group but got `{}`".format(
        library_group.group_name)

    trip_points = TripPoints(
        input_threshold_rise=library_group['input_threshold_pct_rise'] * 0.01,
        input_threshold_fall=library_group['input_threshold_pct_fall'] * 0.01,
        output_threshold_rise=library_group['output_threshold_pct_rise'] * 0.01,
        output_threshold_fall=library_group['output_threshold_pct_fall'] * 0.01,

        slew_lower_threshold_rise=library_group['slew_lower_threshold_pct_rise'] * 0.01,
        slew_upper_threshold_rise=library_group['slew_upper_threshold_pct_rise'] * 0.01,
        slew_lower_threshold_fall=library_group['slew_lower_threshold_pct_fall'] * 0.01,
        slew_upper_threshold_fall=library_group['slew_upper_threshold_pct_fall'] * 0.01,
    )

    return trip_points
