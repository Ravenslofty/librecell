from setuptools import setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='librecell',
      version='0.0.9',
      description='Meta-package for the LibreCell suite.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cmos cell generator layout characterization vlsi asic',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
          'Programming Language :: Python :: 3'
      ],
      url='https://codeberg.org/tok/librecell',
      author='T. Kramer',
      author_email='code@tkramer.ch',
      license='',  # ???
      install_requires=[
          'librecell-common==0.0.9',
          'librecell-layout==0.0.9',
          'librecell-lib==0.0.9',
      ],
      zip_safe=False)
