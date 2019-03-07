from setuptools import setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='lclayout-util',
      version='0.0.1',
      description='Common utility functions for lclayout suite.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cmos cell vlsi asic',
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
      install_requires=[
          'numpy',  # BSD
          'sympy',  # BSD
          'networkx',  # BSD
          'pyparsing',  # MIT
          'lark-parser',  # MIT
          'pyspice',  # GPLv3
          'scipy',  # BSD
          'liberty-parser',  # LGPL
      ],
      zip_safe=False)
