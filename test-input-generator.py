#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:24:25 2019

@author: Matt Caswell

Code to generate inputs for our fixed point DNN accelerator
"""
from keras.datasets import cifar10
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np

datatype = 'int16'
num_int_bits = 8
num_frac_bits = 8

file_name_weight = 'filters.mem' 
file_name_bias = 'bias.mem'
file_name_image = 'input.mem'
file_name_output = 'output.mem'


model_location= "keras_cifar10_trained_model.h5"
model_loaded=tf.keras.models.load_model(model_location) 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#get params of first conv layer    
for layers in model_loaded.layers:
    if layers.get_config()['name'].find('conv')!=-1:
        first_layer=layers
        break
first_layer_params=first_layer.get_weights();

# Get weights and biases of first layer ( -1 to 1 values in float32)
bias=first_layer_params[1]
weight=first_layer_params[0]

# Get a test image (uint8)
image = x_test[1]

# Get the correct output on the same test image
layer_name = 'conv2d_1'
intermediate_layer_model = Model(inputs=model_loaded.input,outputs=model_loaded.get_layer(layer_name).output)
output = intermediate_layer_model.predict(np.expand_dims(image ,0))

# Scale all values to use the fractional bits of the hardware
cbias   = bias   * (2 ** (num_frac_bits ))
cweight = weight * (2 ** (num_frac_bits))
cimage  = image.astype('int16') * (2 ** (num_frac_bits - 1))

# Convert bias, weight, and image to the correct datatype
cbias   = cbias.astype('int16')
cweight = cweight.astype('int16')
cimage  = cimage.astype('int16')
coutput = output.astype('int16') / (29.509 * 7.388)

# Write bias, weight, and image to files
#write image
with open(file_name_image,'wb') as f:
    for k in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                f.write(cimage[i,j,k].tobytes())

#write weight
with open(file_name_weight, 'wb') as f:
    for l in range (weight.shape[3]):
        for k in range(weight.shape[2]):
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    f.write(cweight[i,j,k,l].tobytes())
                    
#write bias             
with open(file_name_bias,'wb') as f:
    for i in range(bias.shape[0]):
        f.write(cbias[i].tobytes())
        
#write output
with open(file_name_output,'w') as f:
    for k in range(coutput.shape[3]):
        for i in range(coutput.shape[1]):
            for j in range(coutput.shape[2]):
                coutput[0,i,j,k].tofile(f,sep="\n",format="%s")
                f.write('\n')
                
