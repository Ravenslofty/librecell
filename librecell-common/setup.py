from setuptools import setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='librecell-common',
      version='0.0.1',
      description='Common utility functions for LibreCell suite.',
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
          'networkx',  # BSD
          'pyparsing',  # MIT
          'pyspice',  # GPLv3
      ],
      zip_safe=False)
