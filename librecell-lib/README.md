# LibreCell - Lib
Characterization kit for CMOS cells.

## Getting started

See install instructions in top-project.

### Characterize a cell
The following example determines the input capacitances and timing delays of a combinational cell.

It is assumed that `FreePDK45` is installed in the users home directory.

Required inputs are:
* --liberty: A template liberty file which defines how the cells should be characterized.
* --include: SPICE files or models to be included.
* --spice: A SPICE file which contains the transistor level circuit of the cell (best including extracted parasitic capacitances).
* --cell: Name of the cell to be characterized.
* --output: Output liberty file which will contain the characterization data.

```sh
lclayout --liberty ~/FreePDK45/osu_soc/lib/files/gscl45nm.lib \
	--include ~/FreePDK45/osu_soc/lib/files/gpdk45nm.m \
	--spice ~/FreePDK45/osu_soc/lib/source/netlists/AND2X1.pex.netlist \
	--cell AND2X1 \
	--output /tmp/and2x1.lib
```
