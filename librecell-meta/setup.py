from setuptools import setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(name='librecell',
      version='0.0.1',
      description='Meta-package for the librecell suite.',
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
      install_requires=[
          'librecell-layout',
          'librecell-time',
      ],
      zip_safe=False)
