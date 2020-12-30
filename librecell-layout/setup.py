#
# Copyright 2019-2020 Thomas Kramer.
#
# This source describes Open Hardware and is licensed under the CERN-OHL-S v2.
#
# You may redistribute and modify this documentation and make products using it
# under the terms of the CERN-OHL-S v2 (https:/cern.ch/cern-ohl).
# This documentation is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY,
# INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
# Please see the CERN-OHL-S v2 for applicable conditions.
#
# Source location: https://codeberg.org/tok/librecell
#
from setuptools import setup, find_packages


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='librecell-layout',
      version='0.0.6',
      description='CMOS standard cell layout generator.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cmos cell generator layout klayout vlsi asic',
      classifiers=[
          # 'License :: OSI Approved :: GNU Affero General Public License v3',
          'Development Status :: 3 - Alpha',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
          'Programming Language :: Python :: 3'
      ],
      url='https://codeberg.org/tok/librecell',
      author='T. Kramer',
      author_email='code@tkramer.ch',
      license='OHL-S v2.0',
      packages=find_packages(),
      package_data={'': ['examples/*']},
      entry_points={
          'console_scripts': [
              'lclayout = lclayout.standalone:main',
              # 'drc_cleaner = lclayout.drc_cleaner.standalone:main',
          ]
      },
      install_requires=[
          'librecell-common',
          'toml==0.10.*',
          'klayout==0.26.*',  # GPLv3
          'numpy==1.*',  # BSD
          'networkx==2.5',  # BSD
          'pyspice==1.4.3',  # GPLv3
          'scipy>=1.5.*',  # BSD
          'liberty-parser==0.0.8',  # GPLv3
          'pysmt==0.9.*',  # Apache-2.0
          'z3-solver==4.8.*',  #
      ],
      zip_safe=False)
