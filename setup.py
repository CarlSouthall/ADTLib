from distutils.core import setup
from setuptools import find_packages
import glob

package_data = ['nn/NNFiles/*'] 
scripts = glob.glob('bin/*')               

setup(
  name = 'ADTLib',
  packages=find_packages(exclude=[]), # this must be the same as the name above
  version = '2.0',
  description = 'Automated Drum Trancription Libray',
  author = 'Carl Southall',
  author_email = 'c-southall@live.co.uk',
  license='BSD',
  url = 'https://github.com/CarlSouthall/ADTLib', # use the URL to the github repo
  download_url = 'https://github.com/CarlSouthall/ADTLib', # I'll explain this in a second
  keywords = ['testing', 'logging', 'example'], # arbitrary keywords
  classifiers = ['Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',],
    scripts=scripts,
   
    install_requires=['numpy','scipy','cython','madmom','tensorflow'],
    package_data={'ADTLib': package_data}, 
)
