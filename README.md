# Automatic Drum Transcription Library (ADTLib)

The automatic drum transcription (ADT) library contains open source ADT algorithms to aid other researchers in areas of music information retrieval (MIR).

## License

This library is published under the BSD license which allows redistribution and modification as long as the copyright and disclaimers are contained. The full license information can be found on the [license](https://github.com/CarlSouthall/AutoDrumTranscritpion/blob/master/LICENSE) page. 

## Installation

#### Required Packages

• [numpy](https://www.numpy.org)   
• [scipy](https://www.scipy.org)  
• [cython](https://www.cython.org)  
• [madmom](https://github.com/CPJKU/madmom)  
• [tensorflow](https://www.tensorflow.org/)

The easiest and suggested method to install the libary is to use pip.

     pip install ADTLib

To update the libary use

     pip install --upgrade ADTlib
     
For futher install information see the [install](https://github.com/CarlSouthall/ADTLib/blob/master/install.md) page.


## Algorithms

The algorithms that are currently contained within the libary are:

• ADTBDRNN: Bi-directional architecture outlined in [1]

## Usage

Algorithms contained within the library are both available as functions for use within python and as command line executable programmes.

###Examples 

#### Command line 

    ADTBDRNN DrumFile1.wav DrumFile2.wav


#### Python function

```Python
from ADTLib.models import ADTBDRNN

TrackNames=['DrumFile1.wav','DrumFile2.wav']
out=ADTBDRNN(TrackNames)
```

See the [usage](https://github.com/CarlSouthall/ADTLib/blob/master/usage.md) page for more information.

## References


| **[1]** |                  **[Southall, C., R. Stables, J. Hockman, Automatic Drum Transcription Using Bi-directional Recurrent                    Neural  Networks, Proceedings of the 17th International Society for Music Information Retrieval Conference (ISMIR), 2016.](https://wp.nyu.edu/ismir2016/wp-content/uploads/sites/2294/2016/07/217_Paper.pdf)**|
| :---- | :--- |

##Help

Any questions please feel free to contact me on carl.southall@bcu.ac.uk





