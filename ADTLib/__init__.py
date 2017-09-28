# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:10:50 2017

@author: CarlSouthall
"""



from . import utils
import os

def ADT(filenames,text='yes',tab='yes',save_dir=None):
    location=utils.location_extract()
    Onsets=[]
    for k in filenames:
        specs=utils.spec(k)
        AFs=utils.system_restore(specs,location)
        PP=utils.load_pp_param(location)    
        Peaks=[]
        for j in range(len(AFs)):
                Peaks.append(utils.meanPPmm(AFs[j][:,0],PP[j,0],PP[j,1],PP[j,2]))
        sorted_p=utils.sort_ascending(Peaks)
        if save_dir!=None:
            os.chdir(save_dir)
        if text=='yes':
            utils.print_to_file(sorted_p,k)
                     
        if tab=='yes':
            utils.tab_create([Peaks[2],Peaks[1],Peaks[0]],k)
            
        Onsets.append({'Kick':Peaks[0],'Snare':Peaks[1],'Hihat':Peaks[2]})
    return Onsets
    
            
        
    