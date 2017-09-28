# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:10:50 2017

@author: CarlSouthall
"""

import os
import inspect
import madmom
import numpy as np
import subprocess
from fpdf import FPDF
import ADTLib
import tensorflow as tf
from tensorflow.contrib import rnn

def spec(file):
    return madmom.audio.spectrogram.Spectrogram(file, frame_size=2048, hop_size=512, fft_size=2048,num_channels=1)
    
def meanPPmm(Track,Lambda,mi,ma,hop=512,fs=44100,dif=0.05):

    m=np.mean(Track)*Lambda;
    if ma != 0:
        if m>ma:
            m=ma
    if mi != 0:
        if m<mi:
            m=mi
    TrackNew=np.zeros(len(Track)+2)
    TrackNew[1:len(Track)+1]=Track
    Track=TrackNew
    onsets=[]
    values=[]
    for i in range(len(Track)-2):
        if Track[i+1] > Track[i] and Track[i+1]>=Track[i+2] and Track[i+1] > m:
            onsets=np.append(onsets,i+1)
            values=np.append(values,Track[i+1])
    if len(onsets) >0: 
        onsets=(onsets*hop)/float(fs)
    for i in range(1,len(onsets)):
        if abs(onsets[i]-onsets[i-1])<dif:
            ind=np.argmax(values[i-1:i+1])
            np.delete(onsets,onsets[i-1+ind])
  
    return onsets

def location_extract():
    return os.path.split((inspect.getfile(ADTLib)))[0]

def load_pp_param(save_path):
    cwd=os.getcwd()
    os.chdir(save_path+'/files')
    x=np.load('PPParams.npy')
    os.chdir(cwd)
    return x
    


def tab_create(Onsets,Filename_):
    quantisation_per_beat=4
    bars_per_line=4
    notation=['x','o','o']
    pre_trackname=Filename_.split('/')
    TrackName=pre_trackname[len(pre_trackname)-1].split('.')[0]+' Drum Tab'
    
    subprocess.call(["DBNDownBeatTracker","single","-o","DB.txt",Filename_])

    DBFile=open("DB.txt")
    DBFile=DBFile.read().split('\n')
    DBFile=DBFile[:len(DBFile)-1]
    for i in range(len(DBFile)):
        DBFile[i]=DBFile[i].split('\t')
        DBFile[i]=np.array([float(DBFile[i][0]),int(DBFile[i][1])])
        
    grid=[]
    if len(DBFile)>0:
        max_beat=np.max(np.array(DBFile),0)[1]
        beat_dif=1/float(quantisation_per_beat)
        for i in range(len(DBFile)-1):
            k=np.arange(DBFile[i][0],DBFile[i+1][0],(DBFile[i+1][0]-DBFile[i][0])/float(quantisation_per_beat))
            beat_poss=DBFile[i][1]
            for j in k:
               if beat_poss >= max_beat:
                   beat_poss=0
               grid.append([j,beat_poss])
               beat_poss+=beat_dif
               
        quantisation_per_bar=int(max_beat*quantisation_per_beat)
        
        grid=np.array(grid)
        
        num_bars=np.ceil(grid.shape[0]/float(quantisation_per_bar))
        bar_grid=[]
        bar_start=np.expand_dims(np.transpose(['HH','SD','KD']),1)
        bar_end=np.expand_dims(np.transpose(['|','|','|']),1)
        for i in range(3):
            bar_grid.append(['|'])
            for j in range(quantisation_per_bar):
                    bar_grid[i].append('-')
        
        
        num_lines=np.int(np.floor(num_bars/float(bars_per_line)))
        last_line=num_bars%float(bars_per_line)
        lines=[]
        lines_new=[]
        for i in range(num_lines):
            lines.append(np.concatenate((bar_start,np.tile(bar_grid,int(bars_per_line)),bar_end),1))
            lines_new.append([])
            for j in range(len(lines[i])):
                lines[i][j]=list(lines[i][j])
        
        
        
        if last_line > 0:
            i+=1
            lines.append(np.concatenate((bar_start,np.tile(bar_grid,int(last_line)),bar_end),1))
            lines_new.append([])
            for j in range(len(lines[i])):
                lines[i][j]=list(lines[i][j])
        
                    
        
        
        onset_locations=[]
        onset_line=[]
        onset_bar_location=[]
        onset_tab_location=[]
        
        for i in range(len(Onsets)):
            onset_locations.append([])
            onset_line.append([])
            onset_tab_location.append([])
            onset_bar_location.append([])
            for j in range(len(Onsets[i])):
                onset_locations[i].append(np.argmin(np.abs(grid[:,0]-Onsets[i][j])))
                onset_line[i].append(np.floor(onset_locations[i][j]/(float(quantisation_per_bar*bars_per_line))))
                onset_bar_location[i].append((onset_locations[i][j]-((onset_line[i][j])*quantisation_per_bar*bars_per_line)))
                onset_tab_location[i].append(onset_bar_location[i][j]+2)
                for k in range(bars_per_line-1):
                    if onset_bar_location[i][j]>=(k+1)*quantisation_per_bar:
                        onset_tab_location[i][j]+=1
                lines[int(onset_line[i][j])][i][int(onset_tab_location[i][j])]=notation[i]
            
        lines_new=[]
        
        for i in range(len(lines)):
            lines_new.append([])
            for j in range(len(lines[i])):
                lines_new[i].append(''.join(lines[i][j]))        
        
        
        pdf = FPDF(format='A4')
        pdf.add_page()
        pdf.set_font("Courier", size=12)
        pdf.cell(200, 10, txt=TrackName,ln=1, align="C")
        pdf.set_font("Courier", size=10)
        
        for i in range(len(lines_new)):
            for j in range(len(lines_new[i])):
                pdf.cell(0,3,txt=lines_new[i][j],ln=1,align="C")
            pdf.cell(0,5,txt='',ln=1,align="C")
        pdf.output(pre_trackname[len(pre_trackname)-1].split('.')[0]+'_drumtab.pdf')
        
        
        os.remove("DB.txt")
    else: 
        print('Error: No beat detected')

def sort_ascending(x):
    in_re=[]
    out_re_final=[]
    in_symbols=['KD','SD','HH']
    for j in range(len(x)):
        in_re.append(np.concatenate(((np.expand_dims(x[j],1),np.tile(in_symbols[j],[len(x[j]),1]))),1))
    in_re=np.vstack(in_re)
    sorted_ind=(in_re[:,0]).astype(float).argsort()
    for j in sorted_ind:
        out_re_final.append([in_re[j]])
    out_re_final=np.squeeze(np.array(out_re_final))
    return out_re_final

def print_to_file(onsets,Filename):
    pre_trackname=Filename.split('/')
    f = open(pre_trackname[len(pre_trackname)-1].split('.')[0]+'.ADT.txt', "w")
    for item,item2 in onsets:
        f.write("%.4f \t %s \n" % (float(item), item2))

    f.close()
    

           
class SA:
     
     def __init__(self, training_aug_data=[],training_data=[], training_labels=[], validation_data=[], validation_labels=[], mini_batch_locations=[], network_save_filename=[], minimum_epoch=5, maximum_epoch=10, n_hidden=[20,20], n_classes=2, cell_type='LSTMP', configuration='B', attention_number=2, dropout=0.75, init_method='zero', truncated=1000, optimizer='Adam', learning_rate=0.003 ,display_train_loss='True', display_accuracy='True',save_location=[],output_act='softmax',snippet_length=100,aug_prob=0):         
         self.train_aug=training_aug_data
         self.features=training_data
         self.targ=training_labels
         self.val=validation_data
         self.val_targ=validation_labels
         self.mini_batch_locations=mini_batch_locations
         self.filename=network_save_filename
         self.n_hidden=n_hidden
         self.n_layers=len(self.n_hidden)
         self.cell_type=cell_type
         self.dropout=dropout
         self.configuration=configuration
         self.init_method=init_method
         self.truncated=truncated
         self.optimizer=optimizer
         self.learning_rate=learning_rate
         self.n_classes=n_classes
         self.minimum_epoch=minimum_epoch
         self.maximum_epoch=maximum_epoch
         self.display_train_loss=display_train_loss
         self.num_batch=len(self.mini_batch_locations)
         self.batch_size=self.mini_batch_locations.shape[1]
         self.attention_number=attention_number
         self.display_accuracy=display_accuracy
         self.batch=np.zeros((self.batch_size,self.features.shape[1]))
         self.batch_targ=np.zeros((self.batch_size,self.targ.shape[2]))
         self.save_location=save_location
         self.output_act=output_act
         self.snippet_length=snippet_length
         self.aug_prob=aug_prob
  
     def cell_create(self,scope_name):
         with tf.variable_scope(scope_name):
             if self.cell_type == 'tanh':
                 cells = rnn.MultiRNNCell([rnn.BasicRNNCell(self.n_hidden[i]) for i in range(self.n_layers)], state_is_tuple=True)
             elif self.cell_type == 'LSTM': 
                 cells = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden[i]) for i in range(self.n_layers)], state_is_tuple=True)
             elif self.cell_type == 'GRU':
                 cells = rnn.MultiRNNCell([rnn.GRUCell(self.n_hidden[i]) for i in range(self.n_layers)], state_is_tuple=True)
             elif self.cell_type == 'LSTMP':
                 cells = rnn.MultiRNNCell([rnn.LSTMCell(self.n_hidden[i]) for i in range(self.n_layers)], state_is_tuple=True)
             cells = rnn.DropoutWrapper(cells, input_keep_prob=self.dropout_ph,output_keep_prob=self.dropout_ph) 
         return cells
     
     def weight_bias_init(self):
               
         if self.init_method=='zero':
            self.biases = tf.Variable(tf.zeros([self.n_classes]))           
         elif self.init_method=='norm':
               self.biases = tf.Variable(tf.random_normal([self.n_classes]))           
         if self.configuration =='B':
             if self.init_method=='zero':  
                 self.weights =tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*2, self.n_classes]))
             elif self.init_method=='norm':
                   self.weights = { '1': tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)], self.n_classes])),'2': tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)], self.n_classes]))} 
         if self.configuration =='R':
             if self.init_method=='zero':  
                 self.weights = tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)], self.n_classes]))     
             elif self.init_method=='norm':
                   self.weights = tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)], self.n_classes]))
      
     def attention_weight_init(self,num):
         if num==0:
             self.attention_weights=[tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*4,self.n_hidden[(len(self.n_hidden)-1)]*2]))]
             self.sm_attention_weights=[tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*2,self.n_hidden[(len(self.n_hidden)-1)]*2]))]
         if num>0:
             self.attention_weights.append(tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*4,self.n_hidden[(len(self.n_hidden)-1)]*2])))
             self.sm_attention_weights.append(tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*2,self.n_hidden[(len(self.n_hidden)-1)]*2])))
     def create(self):
       
       tf.reset_default_graph()
       self.weight_bias_init()
       self.x_ph = tf.placeholder("float32", [1, self.batch.shape[0], self.batch.shape[1]])
       self.y_ph = tf.placeholder("float32", self.batch_targ.shape)
       self.seq=tf.constant(self.truncated,shape=[1])
       self.seq2=tf.constant(self.truncated,shape=[1]) 
       self.dropout_ph = tf.placeholder("float32")
       self.fw_cell=self.cell_create('1')
       self.fw_cell2=self.cell_create('2')
       if self.configuration=='R':
           self.outputs, self.states= tf.nn.dynamic_rnn(self.fw_cell, self.x_ph,
                                             sequence_length=self.seq,dtype=tf.float32)
           if self.attention_number >0:
               self.outputs_zero_padded=tf.pad(self.outputs,[[0,0],[self.attention_number,0],[0,0]])
               self.RNNout1=tf.stack([tf.reshape(self.outputs_zero_padded[:,g:g+(self.attention_number+1)],[self.n_hidden[(len(self.n_hidden)-1)]*((self.attention_number)+1)]) for g in range(self.batch_size)])
               self.presoft=tf.matmul(self.RNNout1, self.weights) + self.biases
           else: 
               self.presoft=tf.matmul(self.outputs[0][0], self.weights) + self.biases
       elif self.configuration=='B':
           self.bw_cell=self.cell_create('1')
           self.bw_cell2=self.cell_create('2')
           with tf.variable_scope('1'):
               self.outputs, self.states= tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, self.x_ph,
                                             sequence_length=self.seq,dtype=tf.float32)
                                              
           self.first_out=tf.concat((self.outputs[0],self.outputs[1]),2)
           with tf.variable_scope('2'):
               self.outputs2, self.states2= tf.nn.bidirectional_dynamic_rnn(self.fw_cell2, self.bw_cell2, self.first_out,
                                                 sequence_length=self.seq2,dtype=tf.float32)
           self.second_out=tf.concat((self.outputs2[0],self.outputs2[1]),2)
            
           for i in range((self.attention_number*2)+1):
               self.attention_weight_init(i)
                
            
       
           self.zero_pad_second_out=tf.pad(tf.squeeze(self.second_out),[[self.attention_number,self.attention_number],[0,0]])
#               self.attention_chunks.append(self.zero_pad_second_out[j:j+attention_number*2])
           self.attention_m=[tf.tanh(tf.matmul(tf.concat((self.zero_pad_second_out[j:j+self.batch_size],tf.squeeze(self.first_out)),1),self.attention_weights[j])) for j in range((self.attention_number*2)+1)]
           self.attention_s=tf.nn.softmax(tf.stack([tf.matmul(self.attention_m[i],self.sm_attention_weights[i]) for i in range(self.attention_number*2+1)]),0)
           self.attention_z=tf.reduce_sum([self.attention_s[i]*self.zero_pad_second_out[i:self.batch_size+i] for i in range(self.attention_number*2+1)],0)
           self.presoft=tf.matmul(self.attention_z,self.weights)+self.biases
       if  self.output_act=='softmax':   
           self.pred=tf.nn.softmax(self.presoft)
           self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.presoft, labels=self.y_ph))
       elif self.output_act=='sigmoid':
           self.pred=tf.nn.sigmoid(self.presoft)
           self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.presoft, labels=self.y_ph))
       if self.optimizer == 'GD':
             self.optimize = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
       elif self.optimizer == 'Adam':
             self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost) 
       elif self.optimizer == 'RMS':
             self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost) 
       self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y_ph,1))
       self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
       self.init = tf.global_variables_initializer()
       self.saver = tf.train.Saver()
       self.saver_var = tf.train.Saver(tf.trainable_variables())
       if self.save_location==[]:
           self.save_location=os.getcwd()
            
        
     def locations_create(self,size):
         self.locations=range(size)
         self.dif=size%self.batch_size
         if self.dif>0:
             for i in xrange(self.batch_size-self.dif):
                 self.locations=np.append(self.locations,0)
         self.location_new=np.reshape(self.locations,[-1,self.batch_size])     
         return self.location_new
 
       
     def implement(self,data):
         with tf.Session() as sess:
             self.saver.restore(sess, self.save_location+'/'+self.filename)
             self.test_out=[];
             for i in xrange(len(data)):
                 self.test_len=len(data[i])
                 self.test_locations=self.locations_create(self.test_len)
                 for k in xrange(len(self.test_locations)):
                    for j in xrange(self.batch_size):
                        self.batch[j]=data[i][self.test_locations[k,j]]
                    if k == 0:
                        self.test_out.append(sess.run(self.pred, feed_dict={self.x_ph: np.expand_dims(self.batch,0),self.dropout_ph:1}))                                    
                    elif k > 0:
                        self.test_out_2=sess.run(self.pred, feed_dict={self.x_ph: np.expand_dims(self.batch,0),self.dropout_ph:1})
                        self.test_out[i]=np.concatenate((self.test_out[i],self.test_out_2),axis=0)
                 self.test_out[i]=self.test_out[i][:self.test_len]
                  
         return self.test_out
                
def system_restore(data,save_path):
    ins=['Kick','Snare','Hihat']
    out=[]
    for i in ins:
        if i =='Kick':
            NN=SA([],np.zeros((1,1024)),np.zeros((1,1,2)),mini_batch_locations=np.zeros([1,1000]),network_save_filename=i+'ADTLibAll',save_location=save_path+"/files",cell_type='LSTMP',attention_number=2,n_hidden=[20,20],n_classes=2,truncated=1000,configuration='B',optimizer='GD')
            NN.create()
        NN.filename=i+'ADTLibAll'    
        out.append(NN.implement([data])[0])

    return out      
 
