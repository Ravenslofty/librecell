from lclayout.layout.layers import *

# Physical size of one data base unit in meters.
db_unit = 1e-9

# Scale transistor width.
transistor_channel_width_sizing = 0.7

# GDS2 layer numbers.


my_active = (1, 0)
my_nwell = (2, 0)
my_nwell2 = (2, 1)
my_poly = (3, 0)
my_poly_contact = (4, 0)
my_diff_contact = (5, 0)
my_metal1 = (6, 0)
my_metal1_label = (6, 1)
my_metal1_pin = (6, 2)
my_via1 = (7, 0)
my_metal2 = (8, 0)
my_metal2_label = (8, 1)
my_metal2_pin = (8, 2)
my_abutment_box = (200, 0)

output_map = {
    l_active: my_active,
    l_nwell: [my_nwell, my_nwell2],
    l_poly: my_poly,
    l_poly_contact: my_poly_contact,
    l_diff_contact: my_diff_contact,
    l_metal1: my_metal1,
    l_metal1_label: my_metal1_label,
    l_metal1_pin: my_metal1_pin,
    l_via1: my_via1,
    l_metal2: my_metal2,
    l_metal2_label: my_metal2_label,
    l_metal2_pin: my_metal2_pin,
    l_abutment_box: my_abutment_box
}

# Define how layers can be used for routing.
# Example for a layer that can be used for horizontal and vertical tracks: {'MyLayer1' : 'hv'}
# Example for a layer that can be contacted but not used for routing: {'MyLayer2' : ''}
routing_layers = {
    l_active: '',
    l_poly: 'hv',
    l_metal1: 'hv',
    l_metal2: 'hv',
}

min_spacing = {
    (l_active, l_active): 50,
    (l_nwell, l_nwell): 50,
    (l_poly, l_nwell): 50,
    (l_poly, l_active): 50,
    (l_poly, l_poly): 50,
    (l_metal1, l_metal1): 100,
    (l_metal2, l_metal2): 220,
}

via_layers = {
    (l_metal1, l_active): l_diff_contact,
    (l_metal1, l_poly): l_poly_contact,
    (l_metal1, l_metal2): l_via1
}

# Layer for the pins.
pin_layer = l_metal1

# Power stripe layer
power_layer = l_metal2

# Layers that can be connected/merged without changing the schematic.
# This can be used to resolve spacing/notch violations by just filling the space.
connectable_layers = {l_nwell}

# Standard cell dimensions.
unit_cell_width = 400
unit_cell_height = 2400  # = row height

gate_length = 50  # Width of the gate polysilicon stripe.

gate_extension = 100  # Minimum length a polysilicon gate must overlap the silicon.

# Routing pitch
routing_grid_pitch_x = unit_cell_width // 2

routing_grid_pitch_y = unit_cell_height // 8

grid_offset_x = routing_grid_pitch_x
grid_offset_y = routing_grid_pitch_y // 2

# Width of power rail.
power_rail_width = 360

minimum_gate_width_nfet = 200
minimum_gate_width_pfet = 200

minimum_pin_width = 50

# NW must be larger than RX.
nw2rx_overlap_y = 100
nw2rx_overlap_x = 100

wire_width = {
    l_poly: 100,
    l_metal1: 100,
    l_metal2: 100
}

wire_width_horizontal = {
    l_poly: 100,
    l_metal1: 100,
    l_metal2: 100
}

via_size = {
    l_poly_contact: 80,
    l_diff_contact: 80,
    l_via1: 100
}

# TODO
minimum_width = {
    l_poly: gate_length,
    l_metal1: 100,
    l_metal2: 100
}

minimum_via_enclosure = {
    l_active: 10,
    l_poly: 10,
    l_metal1: 20,
    l_metal2: 20,
}

minimum_notch = {
    l_active: 50,
    l_poly: 50,
    l_metal1: 50,
    l_metal2: 50,
    l_nwell: 50
}

min_area = {
    l_metal1: 100 * 100,
    l_metal2: 100 * 100,
}

# ROUTING #

# Cost for changing routing direction (horizontal/vertical).
# This will avoid creating zig-zag routings.
orientation_change_penalty = 100

# Routing edge weights by layer.
weights_horizontal = {
    l_poly: 2,
    l_metal1: 1,
    l_metal2: 1,
}
weights_vertical = {
    l_poly: 2,
    l_metal1: 1,
    l_metal2: 2,
}

# Via weights.
via_weights = {
    (l_metal1, l_active): 500,
    (l_metal1, l_poly): 500,
    (l_metal1, l_metal2): 400
}

# Enable double vias between layers.
multi_via = {
    (l_metal1, l_poly): 1,
    (l_metal1, l_metal2): 1,
}
