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
from enum import Enum
from typing import Any, Dict, List, Tuple, Set, Union
from itertools import chain

"""
Data structure for generating LEF files.

References:
[LEF/DEF]: LEF/DEF 5.7 Language Reference
"""

FLOAT_FORMAT = "%.8f"


class LefFormattable:

    def format(self):
        return None


class LefStatement(LefFormattable):
    pass


class LefContainer(LefStatement):
    pass


class Direction(Enum):
    INPUT = 1
    OUTPUT = 2
    OUTPUT_TRISTATE = 3
    INOUT = 4
    FEEDTHRU = 5


class Use(Enum):
    SIGNAL = 0
    ANALOG = 1
    POWER = 2
    GROUND = 3
    CLOCK = 4


class Shape(Enum):
    ABUTMENT = 0
    RING = 1
    FEEDTHRU = 2


class Symmetry(LefStatement, Enum):
    X = 1
    Y = 2
    R90 = 4

    def __hash__(self):
        return self.value


class Class(Enum):
    NONE = 0
    CORE = 1
    BUMP = 2


class MacroClass(LefStatement, Enum):
    COVER = 10
    COVER_BUMP = 11
    RING = 20
    BLOCK = 30
    BLOCK_BLACKBOX = 31
    BLOCK_SOFT = 32
    PAD = 40
    PAD_INPUT = 41
    PAD_OUTPUT = 42
    PAD_INOUT = 43
    PAD_POWER = 44
    PAD_SPACER = 45
    PAD_AREAIO = 46
    CORE = 50
    CORE_FEEDTHRU = 51
    CORE_TIEHIGH = 52
    CORE_TIELOW = 53
    CORE_SPACER = 54
    CORE_ANTENNACELL = 55
    CORE_WELLTAP = 56
    ENDCAP_PRE = 61
    ENDCAP_POST = 62
    ENDCAP_TOPLEFT = 63
    ENDCAP_TOPRIGHT = 64
    ENDCAP_BOTTOMLEFT = 65
    ENDCAP_BOTTOMRIGHT = 66

    def format(self):
        return "CLASS {}".format(self.name.replace("_", " ").upper())


class Point(LefFormattable):

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def format(self):
        return "{} {}".format(self.x, self.y)


class Property(LefStatement):

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def format(self):
        return "{} {}".format(self.name.upper(), self.value)


class Foreign(LefStatement):
    """ Reference to the GDSII data structure and its offset relative to the macro.
    """

    def __init__(self, name, offset: Point):
        self.name = name
        self.offset = offset

    def format(self):
        return "{} {}".format(self.name.upper(), self.offset.format())


class Polygon(LefStatement):

    def __init__(self, points: List[Tuple[int, int]]):
        self._points = points

    def format(self) -> str:
        return "POLYGON {}".format(" ".join(
            (FLOAT_FORMAT % x for x in chain(*self._points)))
        )


class Rect(LefStatement):

    def __init__(self, point1: Tuple[int, int], point2: Tuple[int, int]):
        self._points = [point1, point2]

    def format(self) -> str:
        return "RECT {}".format(" ".join(
            (FLOAT_FORMAT % x for x in chain(*self._points)))
        )


class Layer(LefStatement):

    def __init__(self, layer_name: str):
        self.name = layer_name

    def format(self) -> str:
        return "LAYER {}".format(self.name)


class Port(LefContainer):

    def __init__(self, CLASS: Class, geometries: List[Tuple[Layer, List[Union[Rect, Polygon]]]]):
        self.CLASS = CLASS  # Class
        self.geometries = geometries

    def format(self):
        return [
            "PORT",
            [
                self.CLASS,
                [[l, g] for l, g in self.geometries]
            ],
            "END"
        ]


class Pin(LefContainer):
    """
    [LEF/DEF] p. 193.
    """

    def __init__(self, pin_name: str, direction: Direction, use: Use, shape: Shape, port: Port,
                 property: Dict[str, Any]):
        self.name = pin_name
        self.direction = direction
        self.use = use
        self.shape = shape
        self.port = port
        self.property = property  # {property_name: value, ...}

    def format(self):
        return [
            "PIN {}".format(self.name),
            [
                self.direction,
                self.use,
                self.shape,
                self.port.format(),
            ],
            "END {}".format(self.name),
            []  # Empty line.
        ]


class Obstruction(LefContainer):
    """
    Blockage
    """

    def __init__(self, layer: Layer, geometries: List):
        self.layer = layer
        self.geometries = geometries

    def format(self):
        return [
            "OBS",
            [
                self.layer,
                [g for g in self.geometries]
            ],
            "END",
            []
        ]


class Macro(LefContainer):
    """
    [LEF/DEF] p. 172
    """

    def __init__(self,
                 name: str,
                 macro_class: MacroClass,
                 foreign: Foreign,
                 obstructions: List[Obstruction],
                 origin: Point,
                 symmetry: Set[Symmetry],
                 pins: List[Pin],
                 site: str,
                 property: Dict[str, Any] = None,
                 ):
        self.name = name
        self.obstructions = obstructions  # Obstructions (Blockages), [LEF/DEF] p. 192.
        self.macro_class = macro_class
        self.foreign = foreign
        self.origin = origin
        self.symmetry = symmetry
        self.pins = pins
        self.property = property if property is not None else {}
        self.site = site

    def format(self):
        return [
            "MACRO {}".format(self.name.upper()),
            [
                self.macro_class,
                Property("ORIGIN", self.origin.format()),
                Property("FOREIGN", self.foreign.format()),
                Property("SITE", self.site.upper()),
                Property("SYMMETRY", " ".join((s.name.upper() for s in self.symmetry)))
            ],
            self.pins,
            self.obstructions,
            "END {}".format(self.name.upper()),
            [],
        ]


class LibraryLEF(LefContainer):
    def __init__(self,
                 version: str,
                 busbitchars: str = '[]',
                 dividerchar: str = '/',
                 macros: List[Macro] = None
                 ):
        self.version = version
        self.busbitchars = busbitchars
        self.dividerchar = dividerchar
        self.macros = macros if macros is not None else []

    def format(self):
        return [
            Property("VERSION", self.version),
            Property("NAMESCASESENSITIVE", "ON"),
            Property("BUSBITCHARS", '"{}"'.format(self.busbitchars)),
            Property("DIVIDERCHAR", '"{}"'.format(self.dividerchar)),
            self.macros,
            "END LIBRARY"
        ]


def lef_format(lef_obj, indent=-1, indent_char=' ') -> str:
    """ Convert LEF data structure into a string.
    :param lef_obj:
    :param indent:
    :param indent_char:
    :return: LEF formatted as a string.
    """
    i = indent_char * max(0, indent)
    if isinstance(lef_obj, LefContainer):
        return lef_format(lef_obj.format(), indent, indent_char)
    elif isinstance(lef_obj, list):
        return "\n".join((lef_format(e, indent + 1, indent_char) for e in lef_obj))
    elif isinstance(lef_obj, LefStatement):
        return i + "{};".format(lef_obj.format())
    elif isinstance(lef_obj, str):
        return i + lef_obj
    elif isinstance(lef_obj, Enum):
        return i + "{} {};".format(type(lef_obj).__name__.upper(), lef_obj.name.upper())
    else:
        assert False, "Unsupported type: {} ({})".format(type(lef_obj), lef_obj)


def test_lef():
    port = Port(Class.CORE,
                [(Layer('metal1'),
                  [Polygon([(0, 0), (1, 1)]), Polygon([(0, 0), (1, 1)]),
                   Rect((1, 2), (3, 4)), Rect((1, 2), (3, 4))])])

    pin = Pin(pin_name='pinName',
              direction=Direction.INPUT,
              use=Use.SIGNAL,
              shape=Shape.ABUTMENT,
              property={},
              port=port)

    macro = Macro('AND2X8',
                  macro_class=MacroClass.CORE,
                  obstructions=[Obstruction(Layer('metal1'), [Rect((1, 2), (3, 4))])],
                  origin=Point(0, 0),
                  pins=[pin, pin],
                  property={},
                  symmetry={Symmetry.X, Symmetry.Y, Symmetry.R90},
                  site='someSite')

    library = LibraryLEF(version="0.0", macros=[macro])

    s = lef_format(library)

    print(s)


"""
Order of LEF Statements
[VERSION statement]
[BUSBITCHARS statement]
[DIVIDERCHAR statement]
[UNITS statement]
[MANUFACTURINGGRID statement]
[USEMINSPACING statement]
[CLEARANCEMEASURE statement ;]
[PROPERTYDEFINITIONS statement]
[ LAYER (Nonrouting) statement
| LAYER (Routing) statement] ...
[MAXVIASTACK statement]
[VIA statement] ...
#Fixed vias that can be used inside VIARULE
[VIARULE statement] ...
[VIARULE GENERATE statement] ...
[VIA statement] ...
#Generated vias that can reference VIARULE name
[NONDEFAULTRULE statement] ...
[SITE statement] ...
[MACRO statement
[PIN statement] ...
[OBS statement ...]] ...
[BEGINEXT statement] ...
[END LIBRARY]
"""

"""
Library LEF file
[VERSION statement]
[BUSBITCHARS statement]
[DIVIDERCHAR statement]
[VIA statement] ...
[SITE statement]
[MACRO statement
[PIN statement] ...
[OBS statement ...] ] ...
[BEGINEXT statement] ...
[END LIBRARY]
"""
