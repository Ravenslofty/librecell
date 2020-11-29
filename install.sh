#!/bin/bash

# Install library code shared by multiple parts of librecell.
cd librecell-common
python3 setup.py install
cd ..

# Install lclayout
cd librecell-layout
python3 setup.py install
cd ..

# Install lclib
cd librecell-lib
python3 setup.py install
cd ..

#cd librecell-meta
#python3 setup.py install
#cd ..
