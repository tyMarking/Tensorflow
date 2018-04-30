#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 09:14:47 2018

@author: tymarking
"""

import tensorflow as tf

def main():
    const1 = tf.constant(5)
    tf.summary.scalar("Constant1", const1)
    const2 = tf.constant(6)
    tf.summary.scalar("Constant2", const2)
    constSum = const1+const2
    tf.summary.scalar("Sum", constSum)
    sess = tf.Session()
    
    
    
    
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter("tensorboardTests/second")
    tf.global_variables_initializer().run(session=sess)
    
    summary, result = sess.run([merged,constSum])
    test_writer.add_summary(summary)
    
    print(result)
    
    
    
    
    
    
    
    
    
    
    
    
main()