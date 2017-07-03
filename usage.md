#Usage

This file contains infromation regarding using the toolbox.

###Command line

    ADT [-h] [-c {'DrumSolo','DrumMixture','MultiInstrumentMixture'}] [-od {None,dir}] [-o {yes,no}] [-ot {yes,no}] [I [I ...]]   
    
| Flag   | Name           |   Description                                                       | Default setting  |
| ----  |  -------  | ----- |   ------   |   
| -h     |  help             |   displays help file                                              | n/a     |      
| -c     |       context        |  defines the context ('DrumSolo', 'DrumMixture','MultiInstrumentMixture' )                                        | 'DrumSolo'    |                                          
| -od    |   output_dir      |   location output textfiles are saved                            | None | 
| -o      | output_text     | defines whether the output is stored in a textfile or not    | yes |
| -ot    |   output_tab     |   defines whether a tabulature is created and saved to a pdf       | yes|
| I      |   input_file_names|   single or list of wav file names seperated by spaces                     |  n/a |

#####Examples

    ADT Drum.wav
    
Perform ADT on a single audio file. Saves onset times to a text file in the current directory. Creates a drum tabulature and saves it as a pdf in the current directory.
    
    ADT Drum.wav Drum1.wav Drum2.wav

Perform ADT on multiple audio files. Saves onset times to a text file in the current directory. Creates a drum tabulature and saves it as a pdf in the current directory.
    
    ADT -od ~/Desktop -o no -ot yes Drum.wav Drum1.wav Drum2.wav

Perform ADT on multiple audio files. Create a drum tabulature and saves it to the Desktop.
  
###Python function


```Python

Onsets=ADT(filenames, context='DrumSolo',text='yes',tab='yes',save_dir=None)

```
| Name           |   Description                                                       | Default setting  |
|  -------  | ----- |   ------   |   
|       filenames      | Drum.wav files, must be in a list.                                        | n/a     |                                           
|   context     |   defines the context ('DrumSolo', 'DrumMixture','MultiInstrumentMixture' )                           | 'DrumSolo' |
|   text     |   defines whether the output is stored in a textfile or not ('yes','no' )                           | 'yes' |
|   tab  |   defines whether a tabulature is created and saved to a pdf ('yes','no' )                           | 'yes' |
|   save_dir      |   location output textfiles are saved ('None' (saves in current dir), dir)                     | None | 

#####Examples

```Python
from ADTLib import ADT

Filenames=['Drumfile.wav']
Onsets=ADT(Filenames)
```
Perform ADT on a single audio file. Saves onset times to a text file in the current directory. Creates a drum tabulature and saves it as a pdf in the current directory. Returns onset times per instrument.


```Python
from ADTLib import ADT

Filenames=['Drumfile.wav','Drumfile1.wav']
Onsets=ADT(Filenames, context='DrumSolo', text='no', tab='yes', save_dir='~/Desktop')
```
Perform ADT on multiple audio files. Create a drum tabulature and saves it to the Desktop. 



