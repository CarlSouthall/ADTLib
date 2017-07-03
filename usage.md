#Usage

This file contains infromation regarding using the algorithms within the toolbox.

##ADTBDRNN

ADT architecture defined in [1].

Input: wavfiles names along with control parameters

Output: kick, snare, hihat onsets in seconds.

###Command line

    ADT [-h] [-od {None,dir}] [-o {yes,no}] [-ot {yes,no}] [I [I ...]]   
    
| Flag   | Name           |   Description                                                       | Default setting  |
| ----  |  -------  | ----- |   ------   |   
| -h     |  help             |   displays help file                                              | n/a     |                                           
| -od    |   output_dir      |   location output textfiles are saved                            | None | 
| -o      | output_text     | defines whether the output is stored in a textfile or not    | yes |
| -ot    |   output_tab     |   defines whether a tabulature is created and saved to a pdf       | yes|
| I      |   input_file_names|   single or list of wav file names seperated by spaces                     |  n/a |

#####Examples

    ADT Drum.wav
    
    ADT Drum.wav Drum1.wav Drum2.wav
    
    ADT -od ~/Desktop -o no -ot yes Drum.wav Drum1.wav Drum2.wav

  Output ordered by time printed to a text file in current directory and printed in terminal
  
    ADTBDRNN  -os instrument -od Desktop -p no -ot yes Drum.wav DrumFile1.wav DrumFile2.wav
  
  Output ordered by instrument printed to a text file on the desktop.

  
###Python function
```Python
ADTBDRNN(TrackNames, out_sort='time',ret='yes', out_text='no', savedir='current',close_error=0.05,lambd=[9.7,9.9,4.9])
```
| Name           |   Description                                                       | Default setting  |
|  -------  | ----- |   ------   |   
|       TrackNames      | Drum.wav files, must be in a list if more than one.                                        | n/a     |                                           
|   out_sort     |   defines configuration of the output                              | time |
|   savedir      |   location output textfiles are saved                            | current| 
|   ret           |   defines whether the output is returned from the function      | yes|
|   out_text     |   defines whether the ouput is stored in a textfile or not       | no|
|   close_error|     Maximum distance between two onsets without onsets being combined, in seconds.                 |  0.05 |
|   lambd|     Value used for each instrument within the peak picking stage                |  9.7 9.9 9.4 |

#####Examples

```Python
from ADTLib.models import ADTBDRNN

Filenames='Drumfile.wav'
X=ADTBDRNN(Filenames)
```
Output stored in variable X ordered by time.
  
```Python
from ADTLib.models import ADTBDRNN

Filenames=['Drumfile.wav','Drumfile1.wav']
ADTBDRNN(Filenames,out_sort='instrument',ret='no',out_text='yes',savedir='Desktop')
```
Output ordered by instrument printed to a text file on the desktop.



