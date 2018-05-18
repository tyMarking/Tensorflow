# -*- coding: utf-8 -*-
"""
Created on Fri May 18 09:47:44 2018

@author: 136029
"""
import tensorflow as tf
import numpy as np

def main():
    
    
    
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    
    print(eval_data)
    print(train_labels)
    
    
    
    
    
    
    
    
    
    
    
main()