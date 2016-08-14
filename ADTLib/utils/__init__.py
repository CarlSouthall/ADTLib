# -*- coding: utf-8 -*-
"""
@author: CarlSouthall
"""
from __future__ import absolute_import, division, print_function
import scipy.io.wavfile as wav
import numpy as np
import os


def MeanPP(Track,Lambda):
    
    m=np.mean(Track)*Lambda
    onsets=[]
    Track=np.append(Track,0)
    for i in xrange(len(Track)):
        if Track[i]>Track[i-1] and Track[i]>Track[i+1] and Track[i]> m:
            onsets=np.append(onsets,i)

    if len(onsets) > 0:
        onsets=(onsets*512)/float(44100)            
    return onsets  
    
def Wavread(TrackName):
    x=wav.read(TrackName)
    y=x[1]
    if len(y.shape)>1:
        y=np.squeeze(np.sum(y,axis=1)) 
    y=y/float(np.max(abs(y)))
    
    return y

def arrange_output(Inputs,output_sort='time'):
    
    Names=['BD','SD','HH']
   
    Out=list(np.zeros(len(Inputs)))
    Out1=list(np.zeros(len(Inputs)))
    for i in xrange(len(Inputs)): 
   
        Out[i]=list(np.zeros(len(Inputs[i])))
        Out1[i]=list(np.zeros((1,2)))
        for j in xrange(len(Inputs[i])):    
            Out[i][j]=list(np.zeros((len(Inputs[i][j]))))
            for k in xrange(len(Inputs[i][j])):
                Out[i][j][k]=list(np.zeros(2))
                Out[i][j][k][0]=Inputs[i][j][k]
                Out[i][j][k][1]=Names[j]
    
    
            if len(Out[i][j])>1:
                Out1[i]=np.concatenate([Out1[i],Out[i][j]],axis=0)
            
        Out[i]=Out1[i][1:]
               
        if output_sort=='time':
            Out1=np.array(Out[i][:,0],dtype=float)
            Out[i][:,0]=np.array(np.sort(Out1),dtype=str)
            indexs=np.argsort(Out1)    
            out_names=list(Out[i][:,1])
            for j in xrange(len(indexs)):           
                Out[i][j,1]=out_names[indexs[j]]
        
    
    return Out
    
def write_text(X,names,suffix='.ADT.txt',save_dir='current'):
    
    if save_dir != 'current':
        current_dir=os.getcwd()
        os.chdir(save_dir)
        
    for i in xrange(len(names)):
        file = open(names[i]+suffix, "w")       
        for j in xrange(len(X[i])):
            X[i][j][0]=X[i][j][0][0:8]
            item="    ".join(X[i][j])
            file.write("%s\n" % item)
            
    if save_dir != 'current':
        os.chdir(current_dir)        
         
        
        
       