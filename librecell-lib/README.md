# LibreCell - Lib
Characterization kit for CMOS cells.

## Getting started

See install instructions in top-project.

### Characterize a cell

An ready-to-run example can be found in the `examples` folder.
The script `run_example.sh` should characterize the `INVX1` inverter.

The following example determines the input capacitances and timing delays of a combinational cell.

It is assumed that `FreePDK45` is installed in the users home directory.

Required inputs are:
* --liberty: A template liberty file which defines how the cells should be characterized.
* --include: SPICE files or models to be included.
* --spice: A SPICE file which contains the transistor level circuit of the cell (best including extracted parasitic capacitances).
* --cell: Name of the cell to be characterized.
* --output: Output liberty file which will contain the characterization data.

Characterize a single cell:
```sh
lctime --liberty ~/FreePDK45/osu_soc/lib/files/gscl45nm.lib \
	--include ~/FreePDK45/osu_soc/lib/files/gpdk45nm.m \
	--spice ~/FreePDK45/osu_soc/lib/source/netlists/AND2X1.pex.netlist \
	--cell AND2X1 \
	--output /tmp/and2x1.lib
```

Characterize multiple cells in the same run:
```sh
lctime --liberty ~/FreePDK45/osu_soc/lib/files/gscl45nm.lib \
	--include ~/FreePDK45/osu_soc/lib/files/gpdk45nm.m \
	--spice ~/FreePDK45/osu_soc/lib/source/netlists/*.pex.netlist \
	--cell INVX1 AND2X1 XOR2X1 \
	--output /tmp/invx1_and2x1_xor2x1.lib
```


Vizualize the result:
```sh
libertyviz -l /tmp/and2x1.lib --cell AND2X1 --pin Y --related-pin A --table cell_rise
```

### Characterize a cell with differential inputs

Differential inputs can be specified in the liberty template with the `complementary_pin` attribute.
Only the non-inverted pin should appear in the liberty file.

Differential pairs can also be recognized based on their naming. For example if pairs are named with suffixes `_p` for
the non-inverted pin and `_n` for the inverted pin:

```sh
lctime --diff %_p,%_n ...
```

### Merging liberty files
`lctime` will output a liberty file containing only one cell. The `libertymerge` command allows to merge this kind of
output file back into the liberty template.

The following example will take `base_liberty.lib` as a template and update its `cell` entries with the data found in
the liberty files in the `characterization` directory.
```sh
libertymerge -b base_liberty.lib \
    -o output_liberty.lib \
    -u characterization/*.lib
```
This approach allows to run characterization runs of multiple cells independently and in parallel (e.g using `make`).

### Recognize a cell
`lctime` can recognize the boolean function of cells based on the transistor network. Besides combinational functions
also memory-loops can be found and abstracted into latches or flip-flops.
The `sp2bool` command can be used to analyze cells and dump information about their behaviour. This can be useful for debugging and verification.

Example:
```sh
# Analyze a combinational cell. 
sp2bool sp2bool --spice ~/FreePDK45/osu_soc/lib/files/cells.sp --cell NAND2X1

# Analyze a flip-flop with asynchronous set and reset signals.
sp2bool sp2bool --spice ~/FreePDK45/osu_soc/lib/files/cells.sp --cell DFFSR
```

For cells with *differential* inputs the `--diff` argument must be used to specify differential pairs.
