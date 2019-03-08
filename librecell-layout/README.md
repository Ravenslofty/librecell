# LibreCell - Layout
CMOS Standard Cell layout generator.

## Getting started

See install instructions in top-project.

### Generate a layout
Generate a layout from a SPICE netlist which includes the transistor sizes.
```sh
mkdir mylibrary
lclayout --output-dir mylibrary --tech examples/dummy_tech.py --netlist examples/cells.sp --cell AND2X1
```
