from klayout import db
from ..layout.layers import *

import logging

logger = logging.getLogger(__name__)


def extract_netlist(layout: db.Layout, top_cell: db.Cell, reference: db.Netlist) -> db.Netlist:
    """
    Extract a device level netlist of 3-terminal MOSFETs from the cell `top_cell` of layout `layout`.
    :param layout: Layout object.
    :param top_cell: The top cell of the circuit.
    :return: Netlist as a `klayout.db.Circuit` object.
    """

    # Without netlist comparision capabilities.
    l2n = db.LayoutToNetlist(db.RecursiveShapeIterator(layout, top_cell, []))

    # # With netlist comparision capabilities.
    # lvs = db.LayoutVsSchematic(db.RecursiveShapeIterator(layout, top_cell, []))

    def make_layer(layer_name: str):
        return l2n.make_layer(layout.layer(*layermap[layer_name]), layer_name)

    rnwell = make_layer(l_nwell)
    ractive = make_layer(l_active)
    rpoly = make_layer(l_poly)
    # rpoly_lbl = make_layer(l_poly_label)
    rdiff_cont = make_layer(l_diff_contact)
    rpoly_cont = make_layer(l_poly_contact)
    rmetal1 = make_layer(l_metal1)
    rmetal1_lbl = make_layer(l_metal1_label)
    rvia1 = make_layer(l_via1)
    rmetal2 = make_layer(l_metal2)
    rmetal2_lbl = make_layer(l_metal2_label)

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
    # TODO: what if more than 2 metal layers?

    # Inter-layer
    l2n.connect(rpsd, rdiff_cont)
    l2n.connect(rnsd, rdiff_cont)
    l2n.connect(rpoly, rpoly_cont)
    l2n.connect(rpoly_cont, rmetal1)
    l2n.connect(rdiff_cont, rmetal1)
    l2n.connect(rmetal1, rvia1)
    l2n.connect(rvia1, rmetal2)
    # l2n.connect(rpoly, rpoly_lbl)  # attaches labels
    l2n.connect(rmetal1, rmetal1_lbl)  # attaches labels
    l2n.connect(rmetal2, rmetal2_lbl)  # attaches labels

    # Perform netlist extraction
    logger.debug("Extracting netlist from layout")
    l2n.extract_netlist()

    netlist = l2n.netlist()
    netlist.make_top_level_pins()
    netlist.purge()
    netlist.purge_nets()

    assert netlist.top_circuit_count() == 1, "A well formed netlist should have exactly one top circuit."

    # # Compare against reference.
    # cmp = db.NetlistComparer()
    # compare_result = cmp.compare(netlist, reference)
    # logger.info("Netlist comparision result: {}".format(compare_result))
    #
    # if not compare_result:
    #     print(netlist)
    #     print(reference)

    return netlist.dup()


def compare_netlist(extracted: db.Netlist, reference: db.Netlist) -> bool:
    """
    Check if two netlists are equal.
    :param extracted:
    :param reference:
    :return:
    """
    cmp = db.NetlistComparer()
    compare_result = cmp.compare(extracted, reference)
    logger.info("Netlist comparision result: {}".format(compare_result))

    if not compare_result:
        logger.warning("Netlists don't match.")
        print(extracted)
        print(reference)

    return compare_result
