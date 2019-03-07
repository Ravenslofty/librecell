from setuptools import setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='lctime',
      version='0.0.1',
      description='CMOS standard cell characterization kit.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cmos cell characterization vlsi asic',
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
              'libertyviz = lctime.liberty.visualize:main_plot_timing',
              'libertymerge = lctime.liberty.merge:main',
              'lcsize = lctime.transistor_sizing.width_opt:main',
              'lctime = lctime.characterization.standalone:main'
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
