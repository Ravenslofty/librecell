##
## Copyright (c) 2019 Thomas Kramer.
## 
## This file is part of librecell-common 
## (see https://codeberg.org/tok/librecell/src/branch/master/librecell-common).
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.
##
from pyparsing import *
import re

"""
Simple SPICE parser used to read transistor netlists.
"""


class Resistor:
    def __init__(self, name, n1, n2, value=0, params={}):
        self.name = name
        self.n1 = n1
        self.n2 = n2
        self.value = value
        self.params = params

    def __repr__(self):
        return "Resistor(%s, %s, %s, %f, params=%s)" % (self.name, self.n1, self.n2, self.value, self.params)


class Capacitor:
    def __init__(self, name, n1, n2, value=0):
        self.name = name
        self.n1 = n1
        self.n2 = n2
        self.value = value

    def __repr__(self):
        return "Capacitor(%s, %s, %s, %f)" % (self.name, self.n1, self.n2, self.value)


class MOSFET:
    def __init__(self, name, nd, ng, ns, nb, model_name, params={}):
        self.name = name
        self.nd = nd
        self.ng = ng
        self.ns = ns
        self.nb = nb
        self.model_name = model_name
        self.params = params

    def __repr__(self):
        return "MOSFET(%s, %s, %s, %s, %s, %s, params=%s)" % (
            self.name, self.nd, self.ng, self.ns, self.nb, self.model_name, self.params)


class Subckt:
    def __init__(self, name, ports=[], content=[]):
        self.name = name
        self.ports = ports
        self.content = content

    def __repr__(self):
        return "Subckt(%s, %s, %s)" % (self.name, self.ports, self.content)


class Include:
    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return ".include '%s'" % self.path


def parse_spice(inp):
    # Strip away comments but don't change number of lines.
    inp = re.sub(r"^\*.*$", "", inp, flags=re.MULTILINE)

    # Concatenate lines
    inp = re.sub("\n\+", " ", inp, flags=re.MULTILINE)

    # Don't ignore newline.
    ParserElement.setDefaultWhitespaceChars(' \t')

    # Elementary tokens
    beginl = (Word('\n') * (0, None)).suppress()
    endl = (LineEnd() * (1, None)).suppress()

    def parse_float(t):
        scale = 1
        if len(t) > 1:
            suffix = t[1]
            units = {
                'a': 1e-18,
                'f': 1e-15,
                'p': 1e-12,
                'n': 1e-9,
                'u': 1e-6,
                'U': 1e-6,
                'm': 1e-3,
                'k': 1e3,
                'M': 1e6,
                'G': 1e9,
                'T': 1e12
            }

            scale = units.get(suffix, 1)

        return float(t[0]) * scale

    # number = -123.123e-123
    number = (Combine(
        Optional('-') + Word(nums) + Optional('.' + Word(nums)) + Optional(Word('eE') + Optional('-') + Word(nums)))
              + Optional(Word(alphas)).leaveWhitespace()
              ).setParseAction(lambda t: parse_float(t)).setName('float')

    name = Word(alphanums + '_' + '#')
    net = name.setName('net')
    path = CharsNotIn('\n')

    # .end
    end = (beginl + '.end' + Optional('s') + Optional(name)).suppress()

    # Parameter list
    param = (name + Suppress('=') + number).setParseAction(lambda t: (t[0].upper(), t[1]))
    parameter_list = (param * (1, None)).setParseAction(lambda t: dict(list(t)))

    # Components
    resistor = (
            Combine(Word('rR') + name) + net + net + Optional(number) + Optional(parameter_list)
    ).setParseAction(lambda s, l, t: Resistor(*t))

    capacitor = (
            Combine(Word('cC') + name) + net + net + Optional(number) + Optional(parameter_list)
    ).setParseAction(lambda t: Capacitor(*t))

    mosfet = (
            Combine(Word('mM') + name) + net + net + net + net + Optional(name) + Optional(parameter_list)
    ).setParseAction(lambda t: MOSFET(*t))

    # Component
    component = (resistor | capacitor | mosfet) + endl

    # Include
    include = (Suppress('.include') + path + endl).setParseAction(lambda t: Include(*t))

    # subckt start
    subcircuit_def = (
            beginl +
            Suppress('.subckt') +
            name +
            (net * (0, None)).setParseAction(lambda t: [t]) +
            endl
    )

    # subckt body
    subcircuit = Forward()
    subcircuit << subcircuit_def + ((component | subcircuit | include) * (0, None)).setParseAction(
        lambda t: [t]) + end + endl
    subcircuit.setParseAction(lambda t: Subckt(*t))

    netlist = ((subcircuit | include) * (0, None))

    parsed = netlist.parseString(inp, parseAll=True)

    return parsed


def test_spice_parser():
    inp = """
*
*	
.subckt testCircuit in1 in2 out1
*
R1 gnd vdd 4.7 T=270K
R2 gnd vdd 6.3e3
C1 gnd vdd 0.47
* Comment
M1 gnd in1 vdd gnd nmos T=300K
.end
.include blabla
.subckt empty
.end

.subckt inv1 in out vcc gnd
MN1 out in gnd gnd NMOS L=0.35U W=2.0U
MP2 out in vcc vcc PMOS L=0.35U W=4.0U
.end

.subckt container
R1 gnd vdd 4.7
.subckt child a b c
R1 gnd vdd 4.7
.end
.end

.subckt testCircuit2 in1 in2 out1
R1 gnd vdd 4.7
R2 gnd vdd 6.3

C1 gnd vdd 0.47
C2 gnd vdd 0.47555
.end"""

    parsed = parse_spice(inp)

    print(parsed)
