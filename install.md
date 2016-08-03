# Install

This page contains information regarding installing ADTLib. It is suggested to install the package using [pip](https://pypi.python.org/pypi/pip). If pip is not already installed install it using:

     easy_install pip

#### Required packages.

ADTLib requires the following packages to already be installed:

• [numpy](https://www.numpy.org)   
• [scipy](https://www.scipy.org)  
• [cython](https://www.cython.org)   
• [madmom](https://github.com/CPJKU/madmom)  
• [tensorflow](https://www.tensorflow.org/)

If these are already installed install ADTLib using:

     pip install ADTLib

To update the libary use:

     pip install --upgrade ADTlib
     

If you do not already have the required packages the easiest way to install them all is to first install tensorflow using the instructions on the tensorflow github:

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md 

Then install ADTLib using the command above, this should install all remaining dependencies.

If problems occur try installing each package one by one using:

	pip install <package_name>


