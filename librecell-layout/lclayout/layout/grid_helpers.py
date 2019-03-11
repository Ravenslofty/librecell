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
