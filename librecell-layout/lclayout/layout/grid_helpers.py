      
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
