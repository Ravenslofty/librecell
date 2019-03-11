# LibreCell - Layout
CMOS Standard Cell layout generator.

## Getting started

See install instructions in top-project.

### Generate a layout
Generate a layout from a SPICE netlist which includes the transistor sizes:
* --output-dir: Directory which will be used to store GDS and LEF of the cell
* --tech: Python script file containing design rules and technology related data
* --netlist: A SPICE netlist containing the netlist of the cell as a sub circuit (`.subckt`).
* --cell: Name of the cell. Must match the name of the sub circuit in the SPICE netlist.

```sh
mkdir mylibrary
lclayout --output-dir mylibrary --tech examples/dummy_tech.py --netlist examples/cells.sp --cell AND2X1
```

## Adapting design rules
Design rulesi and technology related data need to be encoded in a Python script file as shown in `examples/dummy_tech.py`.
