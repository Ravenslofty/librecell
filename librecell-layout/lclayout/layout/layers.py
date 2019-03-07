import networkx as nx

l_active = 'active'
l_nwell = 'nwell'
l_poly = 'poly'
l_diff_contact = 'diff_contact'
l_poly_contact = 'poly_contact'
l_metal1 = 'metal1'
l_metal1_label = 'metal1_label'
l_metal1_pin = 'metal1_pin'
l_via1 = 'via1'
l_metal2 = 'metal2'
l_metal2_label = 'metal2_label'
l_metal2_pin = 'metal2_pin'
l_abutment_box = 'abutment_box'

layermap = {
    l_nwell: (1, 0),
    l_active: (2, 0),
    l_poly: (3, 0),
    l_diff_contact: (4, 0),
    l_poly_contact: (5, 0),
    l_metal1: (6, 0),
    l_metal1_label: (6, 1),
    l_metal1_pin: (6, 2),
    l_via1: (7, 0),
    l_metal2: (8, 0),
    l_metal2_label: (8, 1),
    l_metal2_pin: (8, 2),
    l_abutment_box: (100, 0)
}

layermap_reverse = {v: k for k, v in layermap.items()}

via_layers = nx.Graph()
via_layers.add_edge(l_active, l_metal1, layer=l_diff_contact)
via_layers.add_edge(l_poly, l_metal1, layer=l_poly_contact)
via_layers.add_edge(l_metal1, l_metal2, layer=l_via1)