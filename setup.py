from distutils.core import setup
from setuptools import find_packages
import glob

package_data = ['files/*'] 
scripts = glob.glob('bin/*')               

setup(
  name = 'ADTLib',
  packages=find_packages(exclude=[]), 
  version = '2.0.1',
  description = 'Automated Drum Trancription Library',
  author = 'Carl Southall',
  author_email = 'carl.southall@bcu.ac.uk',
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
   
    install_requires=['numpy','scipy','cython','madmom','fpdf'],
    package_data={'ADTLib': package_data},
     
)
