#-*- coding: utf-8 -*-
"""
Created on Fri Apr 03 00:48:40 2015

@author: Prem Sai Kumar Reddy Gangana (psreddy@unm.edu)
"""
import numpy as np


def logistic_regression(eachfold,train_examples,test_examples,train_true_label,test_true_label,weight_array,learn_rate,penalty_term,epochs):
    
    """ Method to model the Logistic Regression Algorithm and test the model on 
        each folded data """
        
    # Normalize and append 1's column at the posistion 0 for train and test in each fold    
    train_append_array=np.ones((int(train_examples.shape[0]),1),dtype=float)
    test_append_array=np.ones((int(test_examples.shape[0]),1),dtype=float)
    train_examples_array=(train_examples-train_examples.min(axis=0))/(train_examples.max(axis=0)-train_examples.min(axis=0))
    test_examples_array=(test_examples-train_examples.min(axis=0))/(train_examples.max(axis=0)-train_examples.min(axis=0))
    train_examples_array=np.append(train_append_array,train_examples_array,axis=1)
    test_examples_array=np.append(test_append_array,test_examples_array,axis=1)
    
    # Delata Matrix generation for each fold of training data
    delta_array=np.zeros((int(weight_array.shape[0]),int(train_examples.shape[0])),dtype=float)
    for index in np.arange(int(train_examples.shape[0])):
        delta_array[train_true_label[index],index]=1
    
    # Posterior Probability P(Y|X,W) calculation and Weight_Array W calculation for train data over epochs
    for epsilon in np.arange(epochs,dtype=float):
        learning_rate=learn_rate/(1+(epsilon/epochs))
        train_posterior_prob=posteriorcalculation(weight_array,train_examples_array)
        weight_array=weightcalculation(train_posterior_prob,weight_array,train_examples_array,delta_array,learning_rate,penalty_term)

    # Posterior Probability P(Y|X,W) calulation and performing predictions for testdata
    test_posterior_prob=posteriorcalculation(weight_array,test_examples_array)
    predicted_label=np.argmax(test_posterior_prob,axis=0)
    
    # Accuracy calculation for each fold
    #accuracy=(np.count_nonzero(predicted_label==test_true_label)/(float(test_examples.shape[0])))*100
    accuracy=(((predicted_label==test_true_label).sum())/(float(test_examples.shape[0])))*100
    print "--> Accuarcy for",(eachfold+1),"fold of 10-folds in % : ",accuracy    
        
    return predicted_label
    
def weightcalculation(posterior_prob,weight_array,data_array,delta_array,learning_rate,penalty_term):
   
   """ Method to calculate the weights with given parameters """
   
   # Weight Array calculation using Gradient Descent
   weight_array=weight_array+learning_rate*(np.dot((delta_array-posterior_prob),data_array)-(penalty_term*weight_array))
   
   return weight_array

def posteriorcalculation(weight_array,data_array):
   
   """ Method to calculate the P(Y|X,W) with given weight array and samples array"""
   
   # Posterior Probability calculation using Logistic Regression
   numerator=np.power(np.e,np.dot(weight_array,data_array.transpose()))
   numerator[int(numerator.shape[0])-1,:]=1
   posterior_prob=numerator/numerator.sum(axis=0)
  
   return posterior_prob
