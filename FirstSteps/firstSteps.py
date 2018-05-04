#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 09:14:47 2018

@author: tymarking
"""

import tensorflow as tf

def main():
    """
    const1 = tf.constant(5)
    tf.summary.scalar("Constant1", const1)
    const2 = tf.constant(6)
    tf.summary.scalar("Constant2", const2)
    mult = const1 *const2
    tf.summary.scalar("Product", mult)
    """
    """
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    
    sumNode = a + b
    multNode = a * b
    outNode = multNode/sumNode 
    """
    
#    with tf.name_scope("Variables") as scope:
    w = tf.Variable([0.3], tf.float32)
    tf.summary.tensor_summary("Weight", w)
    b = tf.Variable([-0.3], tf.float32)
    tf.summary.tensor_summary("Bias", b)
    
#    with tf.name_scope('Givens') as scope:
    x = tf.placeholder(tf.float32)
    tf.summary.tensor_summary("Given_inputs", x)
    y = tf.placeholder(tf.float32)
    tf.summary.tensor_summary("Given_output", y)
        
    
    linear_model = w*x+b
    
    tf.summary.tensor_summary("Linear_Model", linear_model)
    error = tf.square(linear_model - y)
    tf.summary.tensor_summary("Error", error)
    loss = tf.reduce_sum(error)
    tf.summary.tensor_summary("Loss", loss)
    
    
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    
    sess = tf.Session()
    
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter("tensorboardTests/graph4/", sess.graph)
    tf.global_variables_initializer().run(session=sess)
    
    for i in range(1000):
        sess.run(train, {x:[1,2,3,4,5], y:[0,-1,-2,-3,-4]})
    print(sess.run(loss, {x:[1,2,3,4,5], y:[0,-1,-2,-3,-4]}))
#    test_writer.add_summary(summary)
    
#    print(result)
    
    
    
    
    
    
    
  
    
    
    
    
main()