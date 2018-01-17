# Usage

This file contains information regarding using the toolbox.

### Command line

    ADT [-h] [-od {None,dir}] [-o {yes,no}] [-ot {yes,no}] [I [I ...]]   
    
| Flag   | Name           |   Description                                                       | Default setting  |
| ----  |  -------  | ----- |   ------   |   
| -h     |  help             |   displays help file                                              | n/a     |                                        
| -od    |   output_dir      |   location output .txt files are saved                            | None | 
| -o      | output_text     | defines whether the output is stored in a .txt file or not    | yes |
| -ot    |   output_tab     |   defines whether a tabulature is created and saved to a pdf       | yes|
| I      |   input_file_names|   single or list of drum.wav file names separated by spaces                     |  n/a |

##### Examples

    ADT Drum.wav
    
Perform ADT on a single audio file. Saves onset times to a .txt file in the current directory. Creates a drum tabulature and saves it as a pdf in the current directory.
    
    ADT Drum.wav Drum1.wav Drum2.wav

Perform ADT on multiple audio files. Saves onset times to a .txt file in the current directory. Creates a drum tabulature and saves it as a pdf in the current directory.
    
    ADT -od ~/Desktop -o no ~/Drum.wav ~/Desktop/Drum1.wav 

Perform ADT on multiple audio files from different directories. Creates a drum tabulature but not a .txt file and saves it to the desktop.

  
### Python function


```Python

Onsets=ADT(filenames, text='yes', tab='yes', save_dir=None, output_act='no')

```
| Name           |   Description                                                       | Default setting  |
|  -------  | ----- |   ------   |   
|       filenames      | Drum.wav files, must be in a list.                                        | n/a     |                                           
|   text     |   defines whether the output is stored in a .txt file or not ('yes','no' )                           | 'yes' |
|   tab  |   defines whether a tabulature is created and saved to a pdf ('yes','no' )                           | 'yes' |
|   save_dir      |   location output .txt files are saved ('None' (saves in current dir), dir)                     | None | 
|   output_act    |   defines whether the activation functions are also output ('yes','no')                     | 'no' | 

##### Examples

```Python
from ADTLib import ADT

Onsets=ADT(['Drum.wav'])
```
Perform ADT on a single audio file. Saves onset times to a .txt file in the current directory. Creates a drum tabulature and saves it as a pdf in the current directory. Returns onset times per instrument.

```Python
from ADTLib import ADT

Onsets=ADT(['Drum.wav', 'Drum1.wav', 'Drum2.wav'])
```
Perform ADT on multiple audio files. Saves onset times to a .txt file in the current directory. Creates a drum tabulature and saves it as a pdf in the current directory. Returns onset times per instrument.

```Python
from ADTLib import ADT

Onsets=ADT('~/Drum.wav', '~/Desktop/Drum1.wav', text='no', save_dir='~/Desktop')
```    
Perform ADT on multiple audio files from different directories. Creates a drum tabulature but not a .txt file and saves it to the desktop. Returns onset times per instrument.

```Python
from ADTLib import ADTDT

Onsets=ADT(['Drum.wav'], output_act='yes')
```    
Perform ADT on a single audio file. Saves onset times to a .txt file in the current directory. Creates a drum tabulature and saves it as a pdf in the current directory. Returns onset times per instrument and activation functions per track, per instrument.



