from setuptools import setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='librecell',
      version='0.0.1',
      description='Layout generator for CMOS standard cells.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cmos cell generator layout klayout',
      classifiers=[
          # 'License :: OSI Approved :: GNU Affero General Public License v3',
          'Development Status :: 3 - Alpha',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
          'Programming Language :: Python :: 3'
      ],
      url='',
      author='T. Kramer',
      author_email='dont@spam.me',
      license='AGPL',  # ???
      entry_points={
          'console_scripts': [
              'librecell = librecell.standalone:main',
              'drc_cleaner = librecell.drc_cleaner.standalone:main',
              'libertyviz = librecell.liberty.vizualize:main_plot_timing',
              'libertymerge = librecell.liberty.merge:main',
              'librecell_size = librecell.transistor_sizing.width_opt:main',
              'librecell_characterize = librecell.characterization.standalone:main'
          ]
      },
      install_requires=[
          'klayout',  # GPLv3
          'numpy',  # BSD
          'sympy',  # BSD
          'networkx',  # BSD
          'pyparsing',  # MIT
          'lark-parser',  # MIT
          'pyspice',  # GPLv3
          'scipy',  # BSD
          'liberty-parser',  # LGPL
          'pysmt'  # Apache-2.0
      ],
      zip_safe=False)
