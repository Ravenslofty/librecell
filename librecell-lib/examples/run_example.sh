#!/bin/bash

# Characterize the INVX1 cell and write the output into invx1.lib.

lctime --liberty invx1_template.lib --include gpdk45nm.m --spice INVX1.pex.netlist --cell INVX1 --output invx1.lib
