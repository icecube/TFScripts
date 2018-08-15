#!/usr/bin/env python

from distutils.core import setup
exec(compile(open('version.py', "rb").read(),
             'version.py',
             'exec'))

setup(name='tfscripts',
      version=__version__,
      description='Collection of TF functions and helpful additions',
      author='Mirco Huennefeld',
      author_email='mirco.huennefeld@tu-dortmund.de',
      url='https://github.com/mhuen/TFScripts',
      packages=['tfscripts', 'tfscripts.hex'],
      install_requires=['numpy', 'matplotlib'],  # and tensorflow
      )
