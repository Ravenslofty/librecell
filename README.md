# LibreCell
LibreCell aims to be a toolbox for automated synthesis of CMOS logic cells.

The project is in a very early stage and not yet ready to be used for production.
Project structure and API might change heavily in near future.

### Getting started

It is recommended to use a Python 'virtual environment' for installing all Python dependencies:
```sh
python3 -m venv my-librecell-env
# Activate the virtual environment
source ./my-librecell-env/bin/activate
```

Install from git:
```sh
git clone https://codeberg.org/tok/librecell.git
cd librecell/librecell-python

python3 setup.py develop
```

Now, check if the command-line scripts are in the current search path:
```sh
librecell --help
```
If this shows the documentation of the `librecell` command, then things are fine. Otherwise, the `PATH` environment variable needs to be updated to include `$HOME/.local/bin`.

```sh
# Instead of executing this line each time it can be added to ~/.bashrc
export PATH=$PATH:$HOME/.local/bin
```

#### Generate a layout
Generate a layout from a SPICE netlist which includes the transistor sizes.
```sh
mkdir mylibrary
librecell --output-dir mylibrary --tech examples/dummy_tech.py --netlist examples/cells.sp --cell AND2X1
```

#### Characterize a cell
The following example determines the input capacitances and timing delays of a combinational cell.

It is assumed that `FreePDK45` is installed in the users home directory.

Required inputs are:
* --liberty: A template liberty file which defines how the cells should be characterized.
* --include: SPICE files or models to be included.
* --spice: A SPICE file which contains the transistor level circuit of the cell (best including extracted parasitic capacitances).
* --cell: Name of the cell to be characterized.
* --output: Output liberty file which will contain the characterization data.

```sh
librecell_characterize --liberty ~/FreePDK45/osu_soc/lib/files/gscl45nm.lib \
	--include ~/FreePDK45/osu_soc/lib/files/gpdk45nm.m \
	--spice ~/FreePDK45/osu_soc/lib/source/netlists/AND2X1.pex.netlist \
	--cell AND2X1 \	
	--output /tmp/and2x1.lib
```
