# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 04:21:33 2015

@author: Prem Sai Kumar Reddy Gangana (psreddy@unm.edu)
"""

import numpy as np
import k_fold_partition as kfp
import fft_data
import mfcc_data
import logistic_regression as lr
import rank_fft_data as rfd
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import matplotlib.pyplot as plt

# Environment options to ease the output viewing
np.set_printoptions(linewidth=150,precision=6) 

# Defining the ground truth and collection of labels
label=np.zeros((600),dtype=int)
label[0:100]=0   #0-classical
label[100:200]=1 #1-country
label[200:300]=2 #2-jazz
label[300:400]=3 #3-metal
label[400:500]=4 #4-pop
label[500:600]=5 #5-rock
genre_list=['classical','country','jazz','metal','pop','rock']

# Collection of Indices for 10-fold cross validation and 
# Initialize the predict label with zeros
train_label,test_label=kfp.k_fold_indices(label)
predict_label=np.zeros((int(label.shape[0])),dtype=int)

# Defining the parameters for the algorithm
learning_rate=0.01
penalty_term=0.001
epochs=300

# Prints the menu with options
print("Please select an option:")
print("1 -> Build a LR Classifier with 1000 FFT Components")
print("2 -> Build a LR Classifier with 120 ranked FFT Components")
print("3 -> Build a LR Classifier with 13 Mel Components\n")

# Takes the input of options
choice=input()
print("")



def print_results(predicted_label):
    
    # Prints the Confusion_Matrix, Classification and the over-all accuracy for 10 folds
    print"\n--> Overall Accuracy for 10-fold cross validation in % is :",accuracy_score(label,predict_label)*100    
    cm=confusion_matrix(label,predict_label)
    print"\n--> Confusion Matrix for 10-fold cross validation : ( X->Predicted label, Y->Actual label ) \n",cm
    print"\n--> Classification Report \n"
    print (classification_report(label,predict_label,target_names=genre_list))
    
    
    # Define different paamters for the plot and the plotting confusion matrix
    print"\n--> Confusion Matrix Plot :\n"
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(genre_list))
    plt.xticks(tick_marks, genre_list, rotation=45)
    plt.yticks(tick_marks, genre_list)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #plt.show(block=False) # Uncomment this line if not using IPython
    
    # Prints the Parameters used for the model
    print"\n--> Parameters used for Logistic Regression Algorithm with Gradient Descent Rule"
    print"\n    learning_rate:",learning_rate,", penalty_term:",penalty_term,", epochs:",epochs  
    
    
if(choice==1):
    # Gets the fft data and saves it in a numpy array
#    fft_Data=np.zeros((600,1000))
#    for index,each in enumerate(genre_list):
#        fft_Data[(index*100):((index*100)+100),:]=fft_data.get_fft_data(each)
#    np.save('../data/fftdata.npy',fft_Data)
    
    # Loads the FFT data from a numpy array
    # Uncomment this line and give proper path if you already have the FFT data
    fft_Data=np.load('../data/fftdata.npy')
    
    # Iterates the LR algorith with GD for 10 folds
    for eachfold in np.arange(10):
        # Initialize the weight array for each fold with 6 classes and 10001 weights
        weight_array=np.zeros((6,int(fft_Data.shape[1])+1),dtype=float)
        # Calls the Logistic Regression algorithm for each fold with different train and test data
        predict_label[test_label[eachfold]]=lr.logistic_regression(eachfold,fft_Data[train_label[eachfold]],fft_Data[test_label[eachfold]],label[train_label[eachfold]],label[test_label[eachfold]],weight_array,learning_rate,penalty_term,epochs)
    
    # Method to print the accuracies,confusionmatrix and results
    print_results(predict_label)
    
    
elif(choice==2):
    # Gets the fft data and saves it in a numpy array
#    fft_Data=np.zeros((600,1000))
#    for index,each in enumerate(genre_list):
#        fft_Data[(index*100):((index*100)+100),:]=fft_data.get_fft_data(each)
#    np.save('../data/fftdata.npy',fft_Data)
    
    # Loads the FFT data from a numpy array
    # Uncomment this line and give proper path if you already have the FFT data
    fft_Data=np.load('../data/fftdata.npy')
    
    ranked_features=rfd.rank_fft_features(genre_list,fft_Data)
    fft_Data_subset=fft_Data[:,ranked_features]
    
    # Iterates the LR algorith with GD for 10 folds
    for eachfold in np.arange(10):
        # Initialize the weight array for each fold with 6 classes and 121 weights
        weight_array=np.zeros((6,int(fft_Data_subset.shape[1])+1),dtype=float)
        # Calls the Logistic Regression algorithm for each fold with different train and test data
        predict_label[test_label[eachfold]]=lr.logistic_regression(eachfold,fft_Data_subset[train_label[eachfold]],fft_Data_subset[test_label[eachfold]],label[train_label[eachfold]],label[test_label[eachfold]],weight_array,learning_rate,penalty_term,epochs)
    
    # Method to print the accuracies,confusionmatrix and results
    print_results(predict_label)
    
elif(choice==3):
#    mfcc_Data=np.zeros((600,13))    
#    for index,each in enumerate(genre_list):
#        mfcc_Data[(index*100):((index*100)+100),:]=mfcc_data.get_mfcc_data(each)
#    np.save('../data/mfccdata.npy',mfcc_Data)
    
    # Loads the MFCC data from a numpy array
    # Uncomment this line and give proper path if you already have the MFCC data
    mfcc_Data=np.load('../data/mfccdata.npy')
    
    # Iterates the LR algorith with GD for 10 folds
    for eachfold in np.arange(10):
        # Initialize the weight array for each fold with 6 classes and 14 weights
        weight_array=np.zeros((6,int(mfcc_Data.shape[1])+1),dtype=float)
        # Calls the Logistic Regression algorithm for each fold with different train and test data
        predict_label[test_label[eachfold]]=lr.logistic_regression(eachfold,mfcc_Data[train_label[eachfold]],mfcc_Data[test_label[eachfold]],label[train_label[eachfold]],label[test_label[eachfold]],weight_array,learning_rate,penalty_term,epochs)
    
    # Method to print the accuracies,confusionmatrix and results
    print_results(predict_label)        
    
else:
    print("Invalid Option, Please try executing the program again and select the proper option!!")


