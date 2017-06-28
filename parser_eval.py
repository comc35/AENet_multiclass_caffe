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

file_class = "./meta_file/class_labels_indices.csv"
#file_train = "./meta_file/balanced_train_segments.csv"
file_train = "./meta_file/notempty_train_segments.csv"
file_val = "./meta_file/notempty_eval_segments.csv"
write_train = "./train5.txt"
write_val = "./eval5.txt"
f2_train = open(write_train, 'w')
f2_val = open(write_val, 'w')
#f1_train = open(file_train, 'r')

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
with open(file_train,'r') as f1_train:
	reader = csv.reader(f1_train)
	count = 0
	y = []
	for row in reader:
		labels = []
		if count < 3:
			assert(1)
		else:	
			img_idx = 0
			f2_train.write(row[0] +'/%04d.png'%img_idx  + "\n")
			for r in xrange(3,len(row)):
				if r == 3:
					rkey = row[r].split('"')[1]
				elif r == len(row)-1 :
					rkey = row[r].split('"')[0]
				else : 
					rkey = row[r]
				labels.append(label2index[rkey])
			#print(labels)
			y.append(labels)	
		count +=1

with open(file_val,'r') as f1_val:
	reader = csv.reader(f1_val)
	count = 0
	z = []
	for row in reader:
		labels = []
		if count < 3:
			assert(1)
		else:	
			img_idx = 0
			f2_val.write(row[0] +'/%04d.png'%img_idx  + "\n")
			for r in xrange(3,len(row)):
				if r == 3:
					rkey = row[r].split('"')[1]
				elif r == len(row)-1 :
					rkey = row[r].split('"')[0]
				else : 
					rkey = row[r]
				labels.append(label2index[rkey])
			#print(labels)
			z.append(labels)	
		count +=1


mult = MultiLabelBinarizer().fit_transform(y)
mulv = MultiLabelBinarizer().fit_transform(z)
#a = np.random.random(size=(10,527))
yyy = np.array(mult)
h5f = h5py.File('data5_train.h5', 'w')
h5f.create_dataset('rabel', data=yyy)
h5f.close()
f2_train.close()


zzz = np.array(mulv)
h5fv = h5py.File('data5_eval.h5', 'w')
h5fv.create_dataset('rabel', data=zzz)
h5fv.close()
f2_val.close()
