# -*- coding: utf-8 -*-
"""
Created on Wed Apr 02 23:11:42 2015

@author: Prem Sai Kumar Reddy Gangana (psreddy@unm.edu)
"""
import numpy as np

def rank_fft_features(genre_list,fft_Data):

############## Approach 1 : Based on Standard Deviations of Data
    std_dev_genre_fft_features=[] 
    
    # Collects the genre data and calculates the standard deviation for each feature genrewise
    for index,each in enumerate(genre_list):
       genre_fft_features=fft_Data[(index*100):((index*100)+100),:]
       std_dev_genre_fft_features.append(genre_fft_features.std(axis=0))
    
    # Standard Deviation for each feature on whole dataset
    std_dev_fft_features=fft_Data.std(axis=0)
    
    # Gets the deviation of each feature per genre with respect to the deviation of whole data    
    diff_std_dev=np.abs(std_dev_genre_fft_features-std_dev_fft_features)
    # Sorts the deviations in ascending order and collects the top 20 per each genre    
    rank_features=diff_std_dev.argsort(axis=1) 
    top_20_per_genre=rank_features[:,0:20]
    return(np.unique(top_20_per_genre))
    
############# Approach 2 : Based on Entropy of Data. Comment above block and Uncomment below block if need to test the below
#     sum_genre_fft_features=[]
#     for index,each in enumerate(genre_list):
#        genre_fft_features=fft_Data[(index*100):((index*100)+100),:]
#        sum_genre_fft_features.append(genre_fft_features.sum(axis=0))
#     
#     sum_genre_fft_features=np.array(sum_genre_fft_features)
#     probability_of_x_given_y=sum_genre_fft_features/sum_genre_fft_features.sum(axis=1).reshape(6,1)
#     
#     probability_of_y=1/6.0
#     probability_of_x=probability_of_x_given_y.sum(axis=0)/6.0
#     
#     entropy_of_x_given_y=np.sum(-(probability_of_y*probability_of_x_given_y*np.log2(probability_of_x_given_y))-(probability_of_y*(1-probability_of_x_given_y)*np.log2(1-probability_of_x_given_y)),axis=0)
#     entropy_of_x=-(probability_of_x*np.log2(probability_of_x))-((1-probability_of_x)*np.log2((1-probability_of_x)))
#     
#     info_gain=entropy_of_x-entropy_of_x_given_y
#     rank_features=info_gain.argsort()[::-1][:120]
#     return(rank_features)
     
    
