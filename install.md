# Install

This page contains information regarding installing ADTLib. It is suggested to install the package using [pip](https://pypi.python.org/pypi/pip). If pip is not already installed install it using.

     easy_install pip

Then install the automated drum transcription libarary using the following

     pip install ADTLib

To update the libary use

     pip install --upgrade ADTlib
     
This should install all of the extra required packages. If errors occur, try installing each package one by one as demonstrated below.

#### Required packages.

• [numpy](https://www.numpy.org)   
• [scipy](https://www.scipy.org)  
• [cython](https://www.cython.org)   
• [madmom](https://github.com/CPJKU/madmom)  
• [tensorflow](https://www.tensorflow.org/)

To install numpy

     pip install numpy

To install scipy
     
     pip install scipy
     
To install cython

     pip install cython
     
To install madmom
     
     pip install madmom
     
madmom requires [nose](http://nose.readthedocs.io/en/latest/) in addition to the above packages, if this does not automatically install with madmom use.

     pip install nose
     
Although you can use

     pip install tensorflow 
     
to install tensorflow, problems sometimes occur. It is suggested to install tensorflow using the instructions on the tensorflow github https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md 

Once you have installed all of those dependencies you can install ADTLib using pip as demonstrated at the top of this page.
     
     

     
     


