# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:31:04 2018

@author: zhaoyuzhi
"""

import tensorflow as tf
from math import isnan
import numpy as np
import pandas as pd
import xlrd
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGESAVEURL_training = "C://Users//zhaoyuzhi//Desktop//CNN//training"      #your training saving image URL
IMAGESAVEURL_validation = "C://Users//zhaoyuzhi//Desktop//CNN//validation"  #your validation saving image URL
training_excel = xlrd.open_workbook('trainingImageList.xls')                #your training xlsx file
training_table = training_excel.sheet_by_index(0)
validation_excel = xlrd.open_workbook('validationImageList.xls')            #your validation xlsx file
validation_table = validation_excel.sheet_by_index(0)
x_data = np.zeros([8000,64,64], dtype = np.float32)                         #input training_data
y_data = np.zeros([8000,2], dtype = np.float32)                             #correct output training_data
x_test = np.zeros([2000,64,64], dtype = np.float32)                         #input testing_data
y_test = np.zeros([2000,2], dtype = np.float32)                             #correct output testing_data

## handle training data
for i in range(8000):
    #get 64*64 numpy matrix of a single image
    imagename = training_table.cell(i,0).value
    image = Image.open(IMAGESAVEURL_training + '\\' + imagename)
    width, height = image.size
    imagedata = image.getdata()
    imagematrix = np.matrix(imagedata, dtype='float32') / 255.0
    imagendarray = np.reshape(imagematrix, (height, width))
    #get 0 & 1
    label1 = training_table.cell(i,1).value
    label2 = training_table.cell(i,2).value
    #put data into tensorflow inputs
    x_data[i,:,:] = imagendarray
    y_data[i,0] = label1
    y_data[i,1] = label2
    
## handle testing data
for i in range(2000):
    #get 64*64 numpy matrix of a single image
    imagename = validation_table.cell(i,0).value
    image = Image.open(IMAGESAVEURL_training + '\\' + imagename)
    width, height = image.size
    imagedata = image.getdata()
    imagematrix = np.matrix(imagedata, dtype='float32') / 255.0
    imagendarray = np.reshape(imagematrix, (height, width))
    #get 0 & 1
    label1 = validation_table.cell(i,1).value
    label2 = validation_table.cell(i,2).value
    #put data into tensorflow inputs
    x_test[i,:,:] = imagendarray
    y_test[i,0] = label1
    y_test[i,1] = label2

## define function for CNN
def weight_variable(shape):
    # Truncated normal distribution function
    # shape is kernel size, insize and outsize
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride = [1,x_movement,y_movement,1]
    # must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pool_22(x):
    # stride = [1,x_movement,y_movement,1]
    # must have strides[0] = strides[3] = 1
    # ksize(kernel size) = [1,length,height,1]
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')

## define CNN
x = tf.placeholder(tf.float32, shape=[None,64,64], name='x')        #input imagematrix_data to be fed
y = tf.placeholder(tf.float32, shape=[None,2], name='y')            #correct output to be fed
keep_prob = tf.placeholder(tf.float32, name='keep_prob')            #keep_prob parameter to be fed

x_image = tf.reshape(x, [-1,64,64,1])

## convolutional layer 1, kernel 5*5, insize 1, outsize 32
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = conv2d(x_image, W_conv1) + b_conv1                        #outsize = batch*64*64*32
a_conv1 = tf.nn.relu(h_conv1)                                       #outsize = batch*64*64*32

## max pooling layer 1
h_pool1 = max_pool_22(a_conv1)                                      #outsize = batch*32*32*32
a_pool1 = tf.nn.relu(h_pool1)                                       #outsize = batch*32*32*32

## convolutional layer 2, kernel 3*3, insize 32, outsize 64
W_conv2 = weight_variable([3,3,32,64])
b_conv2 = bias_variable([64])
h_conv2 = conv2d(a_pool1, W_conv2) + b_conv2                        #outsize = batch*32*32*64
a_conv2 = tf.nn.relu(h_conv2)                                       #outsize = batch*32*32*64

## max pooling layer 2
h_pool2 = max_pool_22(a_conv2)                                      #outsize = batch*16*16*64
a_pool2 = tf.nn.relu(h_pool2)                                       #outsize = batch*16*16*64

## convolutional layer 3, kernel 3*3, insize 64, outsize 128
W_conv3 = weight_variable([3,3,64,128])
b_conv3 = bias_variable([128])
h_conv3 = conv2d(a_pool2, W_conv3) + b_conv3                        #outsize = batch*16*16*128
a_conv3 = tf.nn.relu(h_conv3)                                       #outsize = batch*16*16*128

## max pooling layer 3
h_pool3 = max_pool_22(a_conv3)                                      #outsize = batch*8*8*128
a_pool3 = tf.nn.relu(h_pool3)                                       #outsize = batch*8*8*128

## convolutional layer 4, kernel 3*3, insize 128, outsize 256
W_conv4 = weight_variable([3,3,128,256])
b_conv4 = bias_variable([256])
h_conv4 = conv2d(a_pool3, W_conv4) + b_conv4                        #outsize = batch*8*8*256
a_conv4 = tf.nn.relu(h_conv4)                                       #outsize = batch*8*8*256

## max pooling layer 4
h_pool4 = max_pool_22(a_conv4)                                      #outsize = batch*4*4*256
a_pool4 = tf.nn.relu(h_pool4)                                       #outsize = batch*4*4*256

## flatten layer
x_flat = tf.reshape(a_pool4, [-1,4096])                             #outsize = batch*(4*4*256) = batch*4096

## fully connected layer 1
W_fc1 = weight_variable([4096,2560])
b_fc1 = bias_variable([2560])
h_fc1 = tf.matmul(x_flat, W_fc1) + b_fc1                            #outsize = batch*2560
a_fc1 = tf.nn.relu(h_fc1)                                           #outsize = batch*2560
a_fc1_dropout = tf.nn.dropout(a_fc1, keep_prob)                     #dropout layer 1

## fully connected layer 2
W_fc2 = weight_variable([2560,2])
b_fc2 = bias_variable([2])
h_fc2 = tf.matmul(a_fc1_dropout, W_fc2) + b_fc2                     #outsize = batch*2
a_fc2 = tf.nn.softmax(h_fc2)                                        #outsize = batch*2

## define loss and accuracy
z = tf.clip_by_value(a_fc2, 1e-10, 1.0)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(z)))
train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(a_fc2,1), tf.argmax(y,1))   #validation dataset comparison
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #calculate the mean value
testaccuracy = list(range(125))
cache_validation = np.zeros([2000,2], dtype = np.float32)   

## start training
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # train model
    for i in range(12):                                             #number of iterations:1000*500=2250000, 36 millon images
        for m in range(500):                                        #training process using training data 18000 images
            train_xbatch = x_data[(m*16):(m*16+16),:,:]             #train 16 data every batch, not including m*16+16
            train_ybatch = y_data[(m*16):(m*16+16),:]               #train 16 data every batch, not including m*16+16
            sess.run(train_step, feed_dict = {x:train_xbatch, y:train_ybatch, keep_prob:0.4})
            if m % 125 == 0:
                print('i step:',i,'   m step:',m)
                print('Loss is:', sess.run(cross_entropy, feed_dict = {x:train_xbatch, y:train_ybatch, keep_prob:1}))
    # test trained model
    for j in range(125):
        test_xbatch = x_test[(j*16):(j*16+16),:,:]                  #test 16 data every batch, not including k*100+100
        test_ybatch = y_test[(j*16):(j*16+16),:]                    #test 16 data every batch, not including k*100+100
        testaccuracy[j] = accuracy.eval(feed_dict = {x:test_xbatch, y:test_ybatch, keep_prob:1})
        cache_validation[(j*16):(j*16+16),:] = a_fc2.eval(feed_dict = {x:test_xbatch, y:test_ybatch, keep_prob:1})

validation_out = np.mean(testaccuracy)
print("validation accuracy is:", validation_out)                    #average accuracy
