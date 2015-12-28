# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 02:05:12 2015

@author: Prem Sai Kumar Reddy Gangana (psreddy@unm.edu)
"""

import numpy as np
from scipy.io import wavfile
from scikits.talkbox.features import mfcc

def get_mfcc_data(genre):
    
    """ Method to read the 13 mel features from the path
        of the audio files """
        
    genre_mfcc_data=np.zeros((100,13))
    for index in np.arange(100):
        # Gets the path of each genre specific audio file in 100 iterations
        if(index<=9):
            path="./"+genre+"/"+genre+".0000"+str(index)+".wav"
        else:
            path="./"+genre+"/"+genre+".000"+str(index)+".wav"
            
        # Reads the audio file and calculates the 13 mel components of it    
        ceps,mspec,spec=mfcc(wavfile.read(path)[1])
        genre_mfcc_data[index,:]= np.mean(ceps[int((ceps.shape[0])/10):int(((ceps.shape[0])*9)/10)],axis=0)            
    
    return genre_mfcc_data
