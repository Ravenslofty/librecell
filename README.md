# LibreCell
LibreCell aims to be a toolbox for automated synthesis of CMOS logic cells.

The project is in a very early stage and not yet ready to be used for production.
Project structure and API might change heavily in near future.

LibreCell consists of different sub-projects:
* librecell-layout: Automated layout generator for CMOS standard cells.
* librecell-lib: Characterization kit for CMOS cells and tool for handling liberty files.
* librecell-common: Code that is used across different LibreCell projects such as a netlist parser.

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
cd librecell

# Install librecell-common
cd librecell-common
python3 setup.py develop
cd ..

# Install lclayout
cd librecell-layout
python3 setup.py develop
cd ..

# Install lclib
cd librecell-lib
python3 setup.py develop
cd ..
```

Now, check if the command-line scripts are in the current search path:
```sh
lclayout --help
```
If this shows the documentation of the `lclayout` command, then things are fine. Otherwise, the `PATH` environment variable needs to be updated to include `$HOME/.local/bin`.

```sh
# Instead of executing this line each time it can be added to ~/.bashrc
export PATH=$PATH:$HOME/.local/bin
```

#### Generate a layout
Generate a layout from a SPICE netlist which includes the transistor sizes.
```sh
cd librecell-layout
mkdir /tmp/mylibrary
lclayout --output-dir /tmp/mylibrary --tech examples/dummy_tech.py --netlist examples/cells.sp --cell AND2X1
# Use a GDS viewer such as KLayout to inspect the generated layout file `/tmp/mylibrary/*.gds`
```

