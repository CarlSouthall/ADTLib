# -*- coding: utf-8 -*-
"""
@author: CarlSouthall
ADTBDRNN

"""


from __future__ import absolute_import, division, print_function

import madmom
import tensorflow as tf
import numpy as np
import scipy 
import os
import inspect
import ADTLib.nn as ADTnn
from ADTLib.utils import Wavread, MeanPP, arrange_output, write_text

def ADTBDRNN(TrackNames, out_sort='time',ret='yes', out_text='no', savedir='current',close_error=0.05,lambd=[9.7,9.9,4.9]):
 
  ''' Bi-directional neural network algorithm outlined in:
    
    Southall, C., R. Stables, J. Hockman, Automatic Drum Transcription Using 
    Bi-directional Recurrent Neural Networks, 
    Proceedings of the 17th International Society for Music Information 
    Retrieval Conference (ISMIR), 2016.
    
    For usage help see github.com/CarlSouthall/ADTLib/usage.md
           
    '''   
    Time_Steps=1
    WL=2048
    HS=512
    
    names=list(np.zeros(len(TrackNames)))
    
    Track=list(np.zeros(len(TrackNames)))
    for i in xrange(len(TrackNames)):
        Track[i]=Wavread(TrackNames[i])
        name=TrackNames[i].split('.wav')
        names[i]=name[0]
            
    Frames=list(np.zeros(len(Track)))
    Train=list(np.zeros(len(Track)))
    for j in xrange(len(Track)):
        NFrames=int(np.ceil(len(Track[j])/float(HS)))
        Frames[j]=np.zeros((NFrames,WL))
        for i in xrange(NFrames):
            Frames[j][i]=np.squeeze(madmom.audio.signal.signal_frame(Track[j],i,WL,HS,origin=-HS))
            
        Spectrogram=madmom.audio.spectrogram.spec(madmom.audio.stft.stft(Frames[j],np.hanning(WL), fft_size=WL))
        Train[j]=np.zeros((len(Spectrogram),Time_Steps,len(Spectrogram[0])))
        
        for i in xrange(len(Spectrogram)):
            for k in xrange(Time_Steps):
                if i-k >= 0:
                    Train[j][i][Time_Steps-k-1]=Spectrogram[i-k,:]
                
    Path=os.path.split(inspect.getfile(ADTnn))
    NNPath=Path[0]
    
    Kout=ADTnn.BDRNNRestoreAll(Train,NNPath+'/NNFiles/BDKAll-1000')
    Sout=ADTnn.BDRNNRestoreAll(Train,NNPath+'/NNFiles/BDSAll-1000')
    Hout=ADTnn.BDRNNRestoreAll(Train,NNPath+'/NNFiles/BDHAll-1000')
   
   
    AF=list(np.zeros(len(Track)))
    P=list(np.zeros(len(Track)))
    for j in xrange(len(Track)):
        AF[j]=list([Kout[j][:,0],Sout[j][:,0],Hout[j][:,0]])
        P[j]=list(np.zeros(3))
        for i in xrange(len(AF[j])):
            P[j][i]=MeanPP(AF[j][i],lambd[i])
            x=np.sort(P[j][i])
            peak=[]
            if len(x) > 0:
                peak=np.append(peak,x[0])
                
            for k in xrange(len(x)-1):
                if (x[k+1]-peak[len(peak)-1]) >= close_error:
                    peak=np.append(peak,x[k+1])
                    
            P[j][i]=peak
               
    P=arrange_output(P,output_sort=out_sort)
    
    if out_text == 'yes':
       write_text(P,names,save_dir=savedir)
    for i in xrange(len(P)):
            Pnew=list(np.zeros(2))
            Pnew[0]=np.array(P[i][:,0],dtype=float)
            Pnew[1]=np.array(P[i][:,1],dtype=str)
            P[i]=Pnew
    
    if len(P)==1:
        P=P[0]
        
    if ret=='yes':
        return P
        
        