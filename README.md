# LibreCell
LibreCell aims to be a toolbox for automated synthesis of CMOS logic cells.

LibreCell is structured in multiple sub-projects:
* [librecell-layout](librecell-layout): Automated layout generator for CMOS standard cells.
* [librecell-lib](librecell-lib): Characterization kit for CMOS cells and tool for handling liberty files.
* [librecell-common](librecell-common): Code that is used across different LibreCell projects such as a netlist parser.
* [librecell-meta](librecell-meta): Convinience Python package for easier installation.

The project is in a very early stage and might not yet be ready for productive use.
Project structure and API might change heavily in near future.

### Getting started
LibreCell can be installed using the Python package manager `pip` or directly from the git repository.

#### Dependencies
The following dependencies must be installed manually:
* python3
* ngspice http://ngspice.sourceforge.net/ : SPICE simulator used for cell characterization.
* z3 https://github.com/Z3Prover/z3 : SMT solver.

Optional dependencies (not required for default configuration):
* GLPK https://www.gnu.org/software/glpk : ILP/MIP solver

Depending on your linux distribution this packages can be installed using the package manager.

Example for Arch Linux:
```sh
sudo pacman -S python ngspice z3
```

#### Installing with pip

It is recommended to use a Python 'virtual environment' for installing all Python dependencies:
```sh
# Create a new virtual environment
python3 -m venv my-librecell-env
# Activate the virtual environment
source ./my-librecell-env/bin/activate

pip3 install librecell
```

#### Installing from git
It is recommended to use a Python 'virtual environment' for installing all Python dependencies:
```sh
# Create a new virtual environment
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

### Known issues

#### Reproducibility
You may want to generate standard cells in a fully reproducable manner.
Right now there is some non-determinism in LibreCell that has not been investigated yet.
The current workaround is to set the `PYTHONHASHSEED` environment variable.

```sh
export PYTHONHASHSEED=42
lclayout ...
```

## Contact
```python
"codextkramerych".replace("x", "@").replace("y", ".")
```
