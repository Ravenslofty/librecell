# LibreCell
LibreCell aims to be a toolbox for automated synthesis of CMOS logic cells.

The project is in a very early stage and not yet ready to be used for production.


### Getting started

Install from git:
```sh
git clone https://codeberg.org/tok/librecell.git
cd librecell/librecell-python

python3 setup.py develop --user
```

#### Generate a layout
Generate a layout from a SPICE netlist which includes the transistor sizes.
```sh
mkdir mylibrary
librecell --output-dir mylibrary --tech example/dummy_tech.py --netlist examples/cells.sp --cell AND2X1
```


