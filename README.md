# LibreCell
LibreCell aims to be a toolbox for automated synthesis of CMOS logic cells.

The project is in a very early stage and not yet ready to be used for production.


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


