
#from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import caffe
import progressbar
import h5py
from sklearn import preprocessing

import ipdb

bar = progressbar.ProgressBar()
caffe_root = './caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

model_def = './deploy_single.prototxt'
model_weights = './training/single_iter_20000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR



dataroot = "/mnt/hdd/dataset/audioset/eval_spectrogram_25ms_6frame/"
with open('./data_val_single.txt') as f:
    content = f.readlines()
probs = np.zeros((len(content),527)) 
i = 0
label = []
for row in bar(content):
	img_file = row.split(' ')[0]
	image = caffe.io.load_image(dataroot + img_file)#image.shape = 50 96 3
	transformed_image = transformer.preprocess('data', image) #trainsformed_image = 3 50 96
	net.blobs['data'].data[...] = transformed_image
	net.forward()
	output_prob = net.blobs['score'].data[0] #(527,)
	probs[i] = output_prob
	label.append(int(row.split(' ')[1]))
	i +=1

#with h5py.File('./label_valf6.h5', 'r') as f:
#	label = f['rabel'][()]	#label.shape (1027365, 527)
#label = np.load('./label_valf6_short.npy','r')

#blabel = MultiLabelBinarizer().fit_transform(label) 
lb = preprocessing.LabelBinarizer()
lb.fit(label)
blabel = lb.transform(label)
probi = (probs >= 0.5).astype(float)
acc = (blabel == probi).astype(float)

for i in xrange(probi.shape[1]):
	print(' %d th acc is %f\n'%(i, sum(acc[:,i]) / acc.shape[0]) )

print(' total acc is %f\n'%(sum(sum(acc)) / np.prod(acc.shape))) 
