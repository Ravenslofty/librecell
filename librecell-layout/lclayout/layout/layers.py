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
import networkx as nx

import operator
import klayout.db as db


class LayerOp:
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def eval(self):
        pass

    def __add__(self, other):
        return LayerOp(operator.add, self, other)

    def __or__(self, other):
        return LayerOp(operator.add, self, other)

    def __and__(self, other):
        return LayerOp(operator.iand, self, other)

    def __sub__(self, other):
        return LayerOp(operator.sub, self, other)

    def __xor__(self, other):
        return LayerOp(operator.xor, self, other)


class Leaf(LayerOp):
    def __init__(self, val):
        self.val = val

    def eval(self):
        return self.val


class AbstractLayer(Leaf):

    def __init__(self, layer_num, layer_purpose):
        self.layer_num = layer_num
        self.layer_purpose = layer_purpose

    def eval(self):
        return self.layer_num, self.layer_purpose


def layer(idx: int,
          purpose: int = 0,
          name: str = None) -> AbstractLayer:
    """ Get a handle to a layer by layer number.
    :param idx: GDS layer number or a string of the form '1/0'.
    :param purpose: GDS layer purpose.
    :param name: Name as a string.
    :return: Handle to the layer.
    """

    if name is None:
        name = '{}/{}'.format(idx, purpose)

    # Allow idx to be a string like '1/0'.
    if isinstance(idx, str):
        s = idx.split('/', 2)
        a, b = s
        idx = int(a)
        purpose = int(b)

    return AbstractLayer(idx, purpose, material=material)


class Mask:
    """ Wrapper around db.Region.
    """

    def __init__(self, region: db.Region):
        self.region = region

    def __add__(self, other):
        return Mask(self.region + other.region)

    def __or__(self, other):
        return self + other

    def __and__(self, other):
        return Mask(self.region & other.region)

    def __sub__(self, other):
        m = Mask(self.region - other.region)
        m.material = self.material
        return m

    def __xor__(self, other):
        return Mask(self.region ^ other.region)

    def __hash__(self):
        return hash(self.region)

    def __equal__(self, other):
        return self.region == other.region


def eval_op_tree(cell: db.Cell, op_node: LayerOp) -> Mask:
    """ Recursively evaluate the layer operation tree.
    :param op_node: Operand node or leaf.
    :return: Returns a `Mask` object containing a `pya.Region` of the layer.
    """

    if isinstance(op_node, AbstractLayer):
        (idx, purpose) = op_node.eval()
        layer_index = layout.layer(idx, purpose)

        region = _flatten_cell(cell, layer_index, selection_box=selection_box)
        result = Mask(region)
    else:
        assert isinstance(op_node, LayerOp)
        op = op_node.op
        lhs = eval_op_tree(cell, op_node.lhs)
        rhs = eval_op_tree(cell, op_node.rhs)
        result = op(lhs, rhs)
    result.region.merge()

    return result


l_ndiffusion = 'ndiffusion'
l_pdiffusion = 'pdiffusion'
l_nwell = 'nwell'
l_pwell = 'pwell'
l_poly = 'poly'
l_poly_label = 'poly_label'
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
    l_pwell: (2, 0),
    l_ndiffusion: (3, 0),
    l_pdiffusion: (4, 0),
    l_poly: (5, 0),
    l_diff_contact: (6, 0),
    l_poly_contact: (7, 0),
    l_metal1: (8, 0),
    l_metal1_label: (8, 1),
    l_metal1_pin: (8, 2),
    l_via1: (9, 0),
    l_metal2: (10, 0),
    l_metal2_label: (10, 1),
    l_metal2_pin: (10, 2),
    l_abutment_box: (100, 0)
}

layermap_reverse = {v: k for k, v in layermap.items()}

via_layers = nx.Graph()
via_layers.add_edge(l_ndiffusion, l_metal1, layer=l_diff_contact)
via_layers.add_edge(l_pdiffusion, l_metal1, layer=l_diff_contact)
via_layers.add_edge(l_poly, l_metal1, layer=l_poly_contact)
via_layers.add_edge(l_metal1, l_metal2, layer=l_via1)
