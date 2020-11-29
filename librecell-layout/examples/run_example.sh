#!/bin/bash

mkdir mylibrary
# Generate the layout of an AND2X1 cewll.
lclayout --output-dir mylibrary --tech dummy_tech.py --netlist cells.sp --cell AND2X1 --verbose
