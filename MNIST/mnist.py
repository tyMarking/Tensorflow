# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:57:56 2018

@author: 136029
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import gzip

import appjar

tf.logging.set_verbosity(tf.logging.INFO)


def main():
    #do the main thing
    print("running")
    app = appjar.gui()
    app.addLabel("title", "Welcome to appJar")
    app.setLabelBg("title", "red")
    app.go()
    opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    conf = tf.ConfigProto(gpu_options=opts)
    conf.gpu_options.allow_growth = True
    trainingConfig = tf.estimator.RunConfig(session_config=conf)

    
    
#    inputLayer = tf.reshape(images, [-1, 28, 28, 1])
#    tf.summary.image("input", inputLayer)
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    
    
#    onehot_train_labels = tf.one_hot(indices=tf.cast(train_labels, tf.int32), depth=10)
#    onehot_eval_labels = tf.one_hot(indices=tf.cast(eval_labels, tf.int32), depth=10)
     # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir="/tmp/mnist_2x64conv+1x256+dropout_model",
            config=trainingConfig)
    
    
    # Set up logging for predictions
    """
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)
    """
      
      
    # Train the model
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
    mnist_classifier.train(
            input_fn=train_input_fn,
            steps=10000)
    """hooks=[logging_hook])"""

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
  
    """
    sess = tf.Session()
#    print(sess.run(inputLayer))
    
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter("tensorboardTests/first")
    tf.global_variables_initializer().run(session=sess)
    
    summary, result = sess.run([merged,inputLayer])
    test_writer.add_summary(summary)
    """
    
    
def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
    """
    conv1 = getConv(input_layer, 32, 5)
    pool1 = getPool(conv1, 2)
    conv2 = getConv(pool1, 64, 5)
    pool2 = getPool(conv2, 2)
    pool2_flat = tf.reshape(pool2, [-1,7*7*64])
    dense = getDense(pool2_flat, 1024)
    #i don't really know what this does
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = getDense(dropout, 10)
    """
    conv1 = getConv(input_layer, 64, 5)
    pool1 = getPool(conv1, 2)
    conv2 = getConv(pool1, 64, 5)
    pool2 = getPool(conv2, 2)
    pool2_flat = tf.reshape(pool2, [-1,7*7*64])
#    inFlat = tf.reshape(input_layer, [-1, 28*28])
    dense1 = getDense(pool2_flat, 256)
    dropout = tf.layers.dropout(
      inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

#    dense2 = getDense(dense1, 32)
    logits = getDense(dropout, 10)
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    """ 
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    """
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
      
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
              loss=loss,
              global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    
    
    
   
    
    
    
    
def getConv(input_layer, filters, size):
    with tf.name_scope('convolutional') as scope:
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=filters,
          kernel_size=size,
          padding="same",
          activation=tf.nn.relu,)
    return conv1

def getPool(input_layer, size):
    with tf.name_scope('pool') as scope:
        pool = tf.layers.max_pooling2d(inputs=input_layer, pool_size=2, strides=2)
    return pool

def getDense(input_layer, units):   
    with tf.name_scope('dense') as scope:
        dense = tf.layers.dense(inputs=input_layer, units=units, activation=tf.nn.relu)
    return dense
    
"""
print("Reading MNIST Data")

#read the MNIST data
trainImages = gzip.open("data/train-images-idx3-ubyte.gz", 'rb')
trainLabels = gzip.open("data/train-labels-idx1-ubyte.gz", 'rb')
imagesBytes = []


trainImages.read(16)
trainLabels.read(8)
trainData = []
images = []
labels = []

#should be 60000
for i in range(60000):
    image = []
    for pixle in trainImages.read(784):
        images.append(float(pixle))
        image.append(pixle/255)
    label = trainLabels.read(1)[0]
    labels.append(label)
    trainData.append((image, [label]))


testImages = gzip.open("data/t10k-images-idx3-ubyte.gz", 'rb')
testLabels = gzip.open("data/t10k-labels-idx1-ubyte.gz", 'rb')
imagesBytes = []

#testing data
testImages.read(16)
testLabels.read(8)
testData = []

#should be 10000
for i in range(100):
    image = []
    for pixle in testImages.read(784):
        image.append(pixle/255)
    label = testLabels.read(1)[0]
    testData.append((image, [label]))
print("Finished reading MNIST data")
"""

if __name__ == "__main__":
    main()


