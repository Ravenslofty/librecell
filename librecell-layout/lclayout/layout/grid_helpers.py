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
      
def grid_floor(x, grid_spacing, grid_offset):
  """ Round down to next grid point.
  """
  return (x-grid_offset)//grid_spacing * grid_spacing + grid_offset
  
def grid_ceil(x, grid_spacing, grid_offset):
  """ Round up to next grid point.
  """
  return grid_floor(x + grid_spacing-1, grid_spacing, grid_offset)

def grid_round(x, grid_spacing, grid_offset):
  """ Round to next grid point.
  """
  return grid_floor(x + grid_spacing//2, grid_spacing, grid_offset)
