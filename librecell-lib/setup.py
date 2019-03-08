from setuptools import setup, find_packages


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='librecell-lib',
      version='0.0.2',
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
              'lctime = lclib.characterization.standalone:main'
          ]
      },
      install_requires=[
          # 'lclayout-util',
          'numpy',  # BSD
          'sympy',  # BSD
          'matplotlib',
          'networkx',  # BSD
          'pyspice',  # GPLv3
          'scipy',  # BSD
          'liberty-parser',  # LGPL
      ],
      zip_safe=False)
