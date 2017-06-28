"""
input
csv file containing test folder list and it's class
csv file containing class information

output
train/val image list .txt
train/val label onehot .h5

"""

import os
import csv
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import h5py
import ipdb
import random

file_class = "./meta_file/class_labels_indices.csv"
#file_train = "./meta_file/balanced_train_segments.csv"
file_train = "./meta_file/notempty_train_segments.csv"
file_val = "./meta_file/notempty_eval_segments.csv"
write_train = "./data_trainf6.txt"
write_val = "./data_valf6.txt"
f2_train = open(write_train, 'w')
f2_val = open(write_val, 'w')
#f1_train = open(file_train, 'r')

#frames = 57 #frame 6 

## label to index dict
label2index = {}
with open(file_class,'r') as f1_class:
	reader = csv.reader(f1_class)
	first = 1
	for row in reader:
		if first == 1:
			first = 0
		else:
			label2index[row[1]] = row[0]


##
folders=[]
datadir = '/mnt/hdd/dataset/audioset/train_spectrogram_25ms_6frame/'##
with open(file_train,'r') as f1_train:
	reader = csv.reader(f1_train)
	count = 0
	y = []
	for row in reader:
		labels = []
		if count < 3:
			assert(1)
		else:	
			frames = len(os.listdir(datadir+row[0]))
			for r in xrange(3,len(row)):
				if r == 3:
					rkey = row[r].split('"')[1]
				elif r == len(row)-1 :
					rkey = row[r].split('"')[0]
				else : 
					rkey = row[r]
				labels.append(label2index[rkey])
		   	for fdx in xrange(frames):
			    	#folders.append(row[0]]
				#f2_train.write(row[0] +'/%04d.png '%fdx + '0'  + "\n")
			    	folders.append(row[0] + '/%04d.png'%fdx)
			    	y.append(labels)	
		count +=1

datadir = '/mnt/hdd/dataset/audioset/eval_spectrogram_25ms_6frame/'##
with open(file_val,'r') as f1_val:
	reader = csv.reader(f1_val)
	count = 0
	z = []
	for row in reader:
		labels = []
		if count < 3:
			assert(1)
		else:	
			frames = len(os.listdir(datadir+row[0]))
			for r in xrange(3,len(row)):
				if r == 3:
					rkey = row[r].split('"')[1]
				elif r == len(row)-1 :
					rkey = row[r].split('"')[0]
				else : 
					rkey = row[r]
				labels.append(label2index[rkey])
		   	for fdx in xrange(frames):
				f2_val.write(row[0] +'/%04d.png '%fdx + '0'  + "\n")
				z.append(labels)	
		count +=1


#ipdb.set_trace()
assert(len(folders)==len(y))
seq = random.sample(xrange(len(folders)),len(folders))
yy=[y[s] for s in seq]
ranfol = [folders[s] for s in seq]
for ran in ranfol:
	f2_train.write(ran + ' 0'  + "\n")


mult = MultiLabelBinarizer().fit_transform(yy)
mulv = MultiLabelBinarizer().fit_transform(z)
#a = np.random.random(size=(10,527))
yyy = np.array(mult)
h5f = h5py.File('label_trainf6.h5', 'w')
h5f.create_dataset('rabel', data=yyy)
h5f.close()
f2_train.close()


zzz = np.array(mulv)
h5fv = h5py.File('label_valf6.h5', 'w')
h5fv.create_dataset('rabel', data=zzz)
h5fv.close()
f2_val.close()
