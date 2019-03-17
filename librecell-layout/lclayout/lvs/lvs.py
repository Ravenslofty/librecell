from klayout import db
from ..layout.layers import *

import logging
logger = logging.getLogger(__name__)

def extract_netlist(ly: db.Layout, top_cell: db.Cell):
    l2n = db.LayoutToNetlist(db.RecursiveShapeIterator(ly, top_cell, []))

    # def make_layer(layer_index, layer_name):
    #     layer_num, layer_type = layer_index
    #
    #     if layer_type in [0, 2]:
    #         return l2n.make_layer(ly.layer(layer_num, layer_type), layer_name)
    #     elif layer_type == 1:
    #         return l2n.make_text_layer(ly.layer(layer_num, layer_type), layer_name)
    #     assert False, "Unknown data type: {}".format(layer_type)
    #
    # r = {
    #     layer_name: make_layer(index, layer_name)
    #     for layer_name, index in layers.layermap.items()
    # }

    l = layermap

    rnwell = l2n.make_layer(l[l_nwell], l_nwell)
    ractive = l2n.make_layer(l[l_active], l_active)
    rpoly = l2n.make_polygon_layer(l[l_poly], l_poly)
    rpoly_lbl = l2n.make_text_layer(l[l_poly_label], l_poly_label)
    rdiff_cont = l2n.make_polygon_layer(l[l_diff_contact], l_diff_contact)
    rpoly_cont = l2n.make_polygon_layer(l[l_poly_contact], l_poly_contact)
    rmetal1 = l2n.make_polygon_layer(l[l_metal1], l_metal1)
    rmetal1_lbl = l2n.make_text_layer(l[l_metal1_label], l_metal1_label)
    rvia1 = l2n.make_polygon_layer(l[l_via1], l_via1)
    rmetal2 = l2n.make_polygon_layer(l[l_metal2], l_metal2)
    rmetal2_lbl = l2n.make_text_layer(l[l_metal1_label], l_metal2_label)

    rpactive = ractive & rnwell
    rpgate = rpactive & rpoly
    rpsd = rpactive - rpgate

    rnactive = ractive - rnwell
    rngate = rnactive & rpoly
    rnsd = rnactive - rngate

    l2n.register(rpactive, 'pactive')
    l2n.register(rpgate, 'pgate')
    l2n.register(rpsd, 'psd')

    l2n.register(rnactive, 'nactive')
    l2n.register(rngate, 'ngate')
    l2n.register(rnsd, 'nsd')

    # PMOS transistor device extraction
    pmos_ex = db.DeviceExtractorMOS3Transistor("PMOS")
    l2n.extract_devices(pmos_ex, {"SD": rpsd, "G": rpgate, "P": rpoly})

    # NMOS transistor device extraction
    nmos_ex = db.DeviceExtractorMOS3Transistor("NMOS")
    l2n.extract_devices(nmos_ex, {"SD": rnsd, "G": rngate, "P": rpoly})

    # Define connectivity for netlist extraction

    # Intra-layer
    l2n.connect(rvia1)
    l2n.connect(rpsd)
    l2n.connect(rnsd)
    l2n.connect(rpoly)
    l2n.connect(rdiff_cont)
    l2n.connect(rpoly_cont)
    l2n.connect(rmetal1)
    l2n.connect(rmetal2)

    # Inter-layer
    l2n.connect(rpsd, rdiff_cont)
    l2n.connect(rnsd, rdiff_cont)
    l2n.connect(rpoly, rpoly_cont)
    l2n.connect(rpoly_cont, rmetal1)
    l2n.connect(rdiff_cont, rmetal1)
    l2n.connect(rmetal1, rvia1)
    l2n.connect(rvia1, rmetal2)
    l2n.connect(rpoly, rpoly_lbl)  # attaches labels
    l2n.connect(rmetal1, rmetal1_lbl)  # attaches labels
    l2n.connect(rmetal2, rmetal2_lbl)  # attaches labels

    # Perform netlist extraction
    logger.debug("Extracting netlist from layout")
    l2n.extract_netlist()

    netlist = l2n.netlist()
    netlist.make_top_level_pins()
    netlist.purge()
    netlist.purge_nets()
