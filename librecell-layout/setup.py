from setuptools import setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='lclayout',
      version='0.0.1',
      description='CMOS standard cell generator.',
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
      author_email='dont@spam.me',
      license='AGPL',  # ???
      entry_points={
          'console_scripts': [
              'lclayout = lclayout.standalone:main',
              #'drc_cleaner = lclayout.drc_cleaner.standalone:main',
          ]
      },
      install_requires=[
          #'lclayout-util',
          'klayout',  # GPLv3
          'numpy',  # BSD
          'sympy',  # BSD
          'networkx',  # BSD
          'pyparsing',  # MIT
          'lark-parser',  # MIT
          'pyspice',  # GPLv3
          'scipy',  # BSD
          'liberty-parser',  # LGPL
          'pysmt',  # Apache-2.0
          'pulp' # MIT
      ],
      zip_safe=False)
