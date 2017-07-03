# Automatic Drum Transcription Library (ADTLib)

The automatic drum transcription (ADT) library contains open source ADT algorithms to aid other researchers in areas of music information retrieval (MIR).

## License

This library is published under the BSD license which allows redistribution and modification as long as the copyright and disclaimers are contained. The full license information can be found on the [license](https://github.com/CarlSouthall/AutoDrumTranscritpion/blob/master/LICENSE) page. 

## Installation

#### Required Packages

• [numpy](https://www.numpy.org)   
• [scipy](https://www.scipy.org)   
• [madmom](https://github.com/CPJKU/madmom)  
• [tensorflow](https://www.tensorflow.org/)

The easiest and suggested method to install the library is to use pip.

     pip install ADTLib

To update the library use

     pip install --upgrade ADTlib
     
For further install information see the [install](https://github.com/CarlSouthall/ADTLib/blob/master/install.md) page.

## Usage

Algorithms contained within the library are both available as functions for use within python and as command line executable programmes.

###Examples 

#### Command line 

    ADT DrumFile1.wav DrumFile2.wav


#### Python function

```Python
from ADTLib import ADT

TrackNames=['DrumFile1.wav','DrumFile2.wav']
out=ADT(TrackNames)
```

See the [usage](https://github.com/CarlSouthall/ADTLib/blob/master/usage.md) page for more information.

## References


| **[1]** |                  **[Southall, C., R. Stables, J. Hockman, Automatic Drum Transcription Using Bi-directional Recurrent                    Neural  Networks, Proceedings of the 17th International Society for Music Information Retrieval Conference (ISMIR), 2016.](https://wp.nyu.edu/ismir2016/wp-content/uploads/sites/2294/2016/07/217_Paper.pdf)**|
| :---- | :--- |

##Help

Any questions please feel free to contact me on carl.southall@bcu.ac.uk





