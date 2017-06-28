import sys 
import caffe
import os

from caffe import layers as L
from caffe import params as P
import numpy as np
import ipdb
import os.path as osp

## PARAM
cl = 527
batch_s = 128


source_train = './data_train_single.txt'
source_val = './data_val_single.txt'
train_proto_name = 'net_train_single.prototxt'
val_proto_name = 'net_val_single.prototxt'

def proto_input_train(n):
	n.data, n.label = L.ImageData(source=source_train ,batch_size=batch_s, ntop=2, root_folder = "/mnt/hdd/dataset/audioset/train_spectrogram_25ms_6frame/", include=[dict(phase=0)],shuffle=True)
	'''
	n.silence_label = L.Silence(n.label, ntop=0)
	n.rabel  = L.HDF5Data(source='./label_trainf6.txt',batch_size=batch_s,include=[dict(phase=0)])
	#slicing label
	tmp_tops =  L.Slice(n.rabel, slice_param={'axis':1}, ntop=cl)
	for idx in xrange(cl):
		n.tops['lr%03d'%idx] = tmp_tops[idx]
	'''

def proto_input_val(n):
	n.data, n.label = L.ImageData(source=source_val ,batch_size=batch_s, ntop=2, root_folder = "/mnt/hdd/dataset/audioset/eval_spectrogram_25ms_6frame/", include=[dict(phase=1)],shuffle=True)
	'''
	n.silence_label = L.Silence(n.label, ntop=0)
	n.rabel  = L.HDF5Data(source='./label_valf6.txt',batch_size=batch_s,include=[dict(phase=1)])
	#slicing label
	tmp_tops =  L.Slice(n.rabel, slice_param={'axis':1}, ntop=cl)
	for idx in xrange(cl):
		n.tops['lr%03d'%idx] = tmp_tops[idx]
	'''''

def proto_model(n):
	base_param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
	
	n.conv1a = L.Convolution(n.data, kernel_size=3, stride=1, num_output=64, param=base_param)
	n.batn1a = L.BatchNorm(n.conv1a , use_global_stats=True)
	n.scal1a = L.Scale(n.batn1a , bias_term= True)
	n.relu1a = L.ReLU(n.scal1a , in_place=True)
	
	n.conv1b = L.Convolution(n.relu1a , kernel_size=3, stride=1, num_output=64, param=base_param)
	n.batn1b = L.BatchNorm(n.conv1b , use_global_stats=True)
	n.scal1b = L.Scale(n.batn1b , bias_term= True)
	n.relu1b = L.ReLU(n.scal1b , in_place=True)
	n.pool1  = L.Pooling(n.relu1b, pool=P.Pooling.MAX, kernel_h=1,kernel_w=2,stride_h=1, stride_w=2)

	n.conv2a = L.Convolution(n.pool1, kernel_size=3, stride=1, num_output=128, param=base_param)
	n.batn2a = L.BatchNorm(n.conv2a , use_global_stats=True)
	n.scal2a = L.Scale(n.batn2a , bias_term= True)	
	n.relu2a = L.ReLU(n.scal2a , in_place=True)

	n.conv2b = L.Convolution(n.relu2a , kernel_size=3, stride=1, num_output=128, param=base_param)
	n.batn2b = L.BatchNorm(n.conv2b , use_global_stats=True)
	n.scal2b = L.Scale(n.batn2b , bias_term= True)	
	n.relu2b = L.ReLU(n.scal2b , in_place=True)
	n.pool2  = L.Pooling(n.relu2b, pool=P.Pooling.MAX, kernel_size=2, stride=2)
	
	n.fc5    = L.InnerProduct(n.pool2, num_output=1024, param=base_param)
	n.relu3  = L.ReLU(n.fc5 , in_place=True)
	n.fc55   = L.InnerProduct(n.relu3, num_output=1024, param=base_param)
	n.relu33 = L.ReLU(n.fc55 , in_place=True)
	n.fc6    = L.InnerProduct(n.relu33, num_output=cl, param=base_param)
	
	#n.score  = L.Softmax(n.fc6)
	#n.logp   = L.Softmax(n.score)
	#n.finlos = L.Eltwise(n.logp,n.rable , operation=P.Eltwise.MUL)
	
	#n.silsig = L.Silence(n.score,ntop=0)
	#slicing feature
	#tmp_tops =  L.Slice(n.score, slice_param={'axis':1}, ntop=cl)
	#for idx in xrange(cl):
	#	n.tops['ft%03d'%idx] = tmp_tops[idx]
	
def proto_loss(n):
	'''
	for i in xrange(cl):
		#n.tops['cls%03d'%i]=L.InnerProduct(n.relu3, num_output=2)
		#n.tops['sigl%03d'%i] = L.SoftmaxWithLoss(n.tops['cls%03d'%i], n.tops['lr%03d'%i])
		#n.tops['sig%02d'%i]=L.Sigmoid(n.tops['ft%d'%i])
		n.tops['sigl%03d'%i] = L.SigmoidCrossEntropyLoss(n.tops['ft%03d'%i], n.tops['lr%03d'%i])


	#loss_item = [item for item in n.tops.keys() if 'sigl' in item ]
	#n.finlos = L.Eltwise(n.relu3,n.relu33, operation=P.Eltwise.SUM)
	#loss_layers = [['sig1%03d'%i] for item in n.tops.keys() if 'sigl' in item]
	#loss_layers = [['sigl%03d'%i] for i in xrange(cl)]
	loss_layers = [item for item in n.tops.keys() if 'sigl' in item]
	#print(loss_layers)
	n.finlos = L.Eltwise(*[n.tops[item] for item in loss_layers], operation=P.Eltwise.SUM, loss_weight=1)
	#n.conc = L.Concat(*[n.tops[item] for item in loss_item], axis = 0)
	'''
	#n.loss = L.SigmoidCrossEntropyLoss(n.score, n.rabel)
	n.loss = L.SoftmaxWithLoss(n.fc6, n.label)


def proto_train():
	n = caffe.NetSpec()
	proto_input_train(n)
	proto_model(n)
	proto_loss(n)
	return str(n.to_proto())

def proto_val():
	n = caffe.NetSpec()
	proto_input_val(n)
	proto_model(n)
	proto_loss(n)
	return str(n.to_proto())


with open(osp.join('./', train_proto_name), 'w') as f:
	f.write(proto_train())
with open(osp.join('./', val_proto_name), 'w') as f:
	f.write(proto_val())

