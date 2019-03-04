import klayout.db as pya


def fill_notches(region: pya.Region, minimum_notch: int) -> pya.Region:
    """ Fill notches in a pya.Region.
    :param region:
    :param minimum_notch:
    :return:
    """

    notches = region.notch_check(minimum_notch)
    spaces = region.space_check(minimum_notch)
    notches = list(notches) + list(spaces)
    s = pya.Shapes()
    s.insert(region)
    for edge_pair in notches:
        a, b = edge_pair.first, edge_pair.second
        # Find smaller edge (a)
        a, b = sorted((a, b), key=lambda e: e.length())

        # Construct a minimal box to fill the notch
        box = a.bbox()
        # Extend box of shorted edge by points of longer edge
        box1 = box + b.p1
        box2 = box + b.p2

        # Take the smaller box.
        min_box = min([box1, box2], key=lambda b: b.area())

        s.insert(min_box)

    result = pya.Region(s)
    result.merge()
    return result

