from distutils.core import setup
from setuptools import find_packages
import glob

package_data = ['nn/NNFiles/*'] 
scripts = glob.glob('bin/*')               

setup(
  name = 'ADTLib',
  packages=find_packages(exclude=[]), 
  version = '0.7',
  description = 'Automated Drum Trancription Library',
  author = 'Carl Southall',
  author_email = 'c-southall@live.co.uk',
  license='BSD',
  url = 'https://github.com/CarlSouthall/ADTLib', 
  download_url = 'https://github.com/CarlSouthall/ADTLib', 
  keywords = ['Drums', 'Transcription', 'Automated'],
  scripts=scripts,
  classifiers = [
                 'Programming Language :: Python :: 2.7',
                 'License :: OSI Approved :: BSD License',
                 'License :: Free for non-commercial use',
                 'Topic :: Multimedia :: Sound/Audio :: Analysis',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
   
    install_requires=['numpy','scipy','cython','madmom'],
    package_data={'ADTLib': package_data},
     
)
