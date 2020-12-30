from setuptools import setup, find_packages


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='librecell-common',
      version='0.0.7',
      description='Common utility functions for LibreCell suite.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cmos cell vlsi asic',
      classifiers=[
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Development Status :: 3 - Alpha',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
          'Programming Language :: Python :: 3'
      ],
      url='https://codeberg.org/tok/librecell',
      author='T. Kramer',
      author_email='code@tkramer.ch',
      license='GPLv3',
      packages=find_packages(),
      install_requires=[
          'networkx==2.5',  # BSD
          'pyspice==1.4.3',  # GPLv3
      ],
      zip_safe=False)
