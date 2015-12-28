# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 04:05:44 2015

@author: Prem Sai Kumar Reddy Gangana (psreddy@unm.edu)
"""
import numpy as np

def k_fold_indices(label):
    
    """ Method to get the 10-fold partition of examples from the path
        of the audio files """
    
    # Initializes the empty lists        
    train_label=[]
    test_label=[]
    
    # Genertaes the array of numbers ranging from 1 to number fo examples        
    kfold=np.arange(len(label))+1
    
    # Iterate and append the indices of traindata and test data
    for each in np.arange(10):
        test_label.append((kfold%10)==each)        
        train_label.append((kfold%10)!=each)
           
    return train_label,test_label
    
