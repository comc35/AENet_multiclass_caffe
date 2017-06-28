'''
re write csv file
some folder has no file
some folder in list doesn't exist

'''

import os
import csv
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import h5py

file_class = "./meta_file/class_labels_indices.csv"
file_train = "./meta_file/eval_segments.csv"
rewrite_train = "./meta_file/notempty_eval_segments.csv"
w = open(rewrite_train, 'w')
writer = csv.writer(w, lineterminator = "\n")
#f1_train = open(file_train, 'r')

foldername = {}
datadir = '/mnt/hdd/dataset/audioset/eval_spectrogram_25ms_6frame/'
aa = os.listdir(datadir)

##
with open(file_train,'r') as f1_train:
	reader = csv.reader(f1_train)
	count = 0
	y = []
	trct = 0
	for row in reader:
		labels = []
		if count < 3:
			#print(row)
			#f2_train.writerow(row)
			writer.writerow(row)
		#elif count ==14:
		else:	
			if row[0] in aa:
				if len(os.listdir(datadir+row[0])) !=0:
					writer.writerow(row)
				#f2_train.write(row)
					trct +=1


		count +=1
w.close()
