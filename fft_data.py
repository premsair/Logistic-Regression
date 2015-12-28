"""
Created on Sun Mar 29 02:05:12 2015

@author: Prem Sai Kumar Reddy Gangana (psreddy@unm.edu)
"""

import numpy as np
import scipy as sp
from scipy.io import wavfile

def get_fft_data(genre):

    """ Method to read the 1000 fft features from the path
        of the audio files """

    genre_fft_data=np.zeros((100,1000))
    for index in np.arange(100):    
        # Gets the path of each genre specific audio file in 100 iterations
        if(index<=9):
            path="./"+genre+"/"+genre+".0000"+str(index)+".wav"
        else:
            path="./"+genre+"/"+genre+".000"+str(index)+".wav"
        # Reads the audio file and takes first 1000 fft components of it
        sample_rate,X=wavfile.read(path)    
        genre_fft_data[index,:]= abs(sp.fft(X)[:1000])
    
    return genre_fft_data
