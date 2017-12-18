# Automatic Drum Transcription Library (ADTLib)

The automatic drum transcription (ADT) library contains open source ADT algorithms to aid other researchers in areas of music information retrieval (MIR). The algorithms return both a .txt file of kick drum, snare drum, and hi-hat onsets and an automatically generated drum tabulature. 

### Browser Version (ADTWeb)

[ADTWeb](http://dmtlab.bcu.ac.uk/ADT/): a browser based version of ADTLib is now available. 

## License

This library is published under the BSD license which allows redistribution and modification as long as the copyright and disclaimers are contained. The full license information can be found on the [license](https://github.com/CarlSouthall/ADTLibNew/blob/master/LICENSE.txt) page. 

## Installation

#### Required Packages

• [numpy](https://www.numpy.org)   
• [scipy](https://www.scipy.org)   
• [madmom](https://github.com/CPJKU/madmom)   
• [tensorflow](https://www.tensorflow.org/)  
• [fpdf](https://pyfpdf.readthedocs.io/en/latest/) (for tab creation)

The easiest and suggested method to install the library is to use pip:

     pip install ADTLib

To update the library use:

     pip install --upgrade ADTLib
     
## Usage

Algorithms contained within the library are both available as functions for use within python and as command line executable programmes.

### Examples 

#### Command line 

    ADT Drum.wav


#### Python function

```Python
from ADTLib import ADT

out=ADT(['Drum.wav'])
```

See the [usage](https://github.com/CarlSouthall/ADTLibNew/blob/master/usage.md) page for more information.

## References


| **[1]** |                  **[Southall, C., R. Stables, J. Hockman, Automatic Drum Transcription Using Bi-directional Recurrent                    Neural  Networks, Proceedings of the 17th International Society for Music Information Retrieval Conference (ISMIR), 2016.](https://carlsouthall.files.wordpress.com/2017/12/ismiradt2016.pdf)**|
| :---- | :--- |

| **[2]** |                  **[Southall, C., R. Stables, J. Hockman, Automatic Drum Transcription For Polyphonic Recordings Using Soft Attention Mechanisms and Convolutional Neural Networks, Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR), 2017.](https://carlsouthall.files.wordpress.com/2017/12/ismir2017adt.pdf)**|
| :---- | :--- |

| **[3]** |                  **[Southall, C., N. Jillings, R. Stables, J. Hockman, ADTWeb: An Open Source Browser Based Automatic Drum Transcription System. Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR), 2017.](https://carlsouthall.files.wordpress.com/2017/12/ismir2017adtweb.pdf)**|
| :---- | :--- |

## Help

Any questions please feel free to contact me on carl.southall@bcu.ac.uk





