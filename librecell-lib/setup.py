#
# Copyright (c) 2019-2020 Thomas Kramer.
#
# This file is part of librecell 
# (see https://codeberg.org/tok/librecell).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
from setuptools import setup, find_packages


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='librecell-lib',
      version='0.0.3.post0',
      description='CMOS standard cell characterization kit.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cmos cell characterization vlsi asic',
      classifiers=[
          'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
          'Development Status :: 3 - Alpha',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
          'Programming Language :: Python :: 3'
      ],
      url='https://codeberg.org/tok/librecell',
      author='T. Kramer',
      author_email='dont@spam.me',
      license='AGPL',
      packages=find_packages(),
      package_data={'': ['examples/*', 'test_data/*']},
      entry_points={
          'console_scripts': [
              'libertyviz = lclib.liberty.visualize:main_plot_timing',
              'libertymerge = lclib.liberty.merge:main',
              'lcsize = lclib.transistor_sizing.width_opt:main',
              'lctime = lclib.characterization.main_lctime:main'
          ]
      },
      install_requires=[
          'librecell-common',
          'numpy',  # BSD
          'sympy',  # BSD
          'matplotlib',
          'networkx',  # BSD
          'pyspice',  # GPLv3
          'scipy',  # BSD
          'liberty-parser',  # GPLv3
          'joblib'
      ],
      zip_safe=False)
