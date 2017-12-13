import numpy as np
import cv2
import glob
import os,errno
from sklearn.cluster import KMeans
from shutil import copyfile
import pickle
import math
from pathlib import Path
import shutil
q1 = '/home/nfs/shubham9/test/proj/lavisha/result'
if os.path.isdir(q1):
	shutil.rmtree(q1)



data_path = '/home/nfs/shubham9/test/proj/lavisha/101categories'
#query_path = '/home/lavisha/Downloads/cs445_Project/query'
result_path = '/home/nfs/shubham9/test/proj/lavisha/result/'
data_dir = ''.join([data_path, '/*.jpg'])
#query_dir = ''.join([query_path, '/*.jpg'])
list1 = os.listdir(data_path) 


datatrain = np.load('datatrain.npy')
datatest = np.load('datatest.npy')
max_feature = 500
no_result = 1



no_images = len(datatrain)
tfidf = []
feature_dim = 128
# Determining the clusterno of a keypoint
def findcluster(pt,n2):	
	i = n2.val.predict(pt)[0]
	#print("into cl:",i)
	if(n2.children[i].leaf!=-1):
		 return n2.children[i].leaf
	else:
		return findcluster(pt,n2.children[i])

class node:
	def __init__(self):
		self.val = None
		self.leaf = -1
		self.children = []

ktree_filename = 'ktree.pkl'
tfidf = np.load('tfidf.npy')
imgdic = np.load('imgdic.npy')
ktree_load_pkl = open(ktree_filename, 'rb')
n1 = pickle.load(ktree_load_pkl)
print("4. For query images")
sift = cv2.xfeatures2d.SIFT_create()
total = 0
correct = 0
for filename in datatest:
	img = cv2.imread(filename)
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	kp, des = sift.detectAndCompute(gray,None)
	no_key = min(max_feature,len(des))	
	key1 = des[0:no_key,:]
	qdic = {}
	visited = {}
	for i in range(no_key):
		pt = key1[i] 
		pt1 = np.reshape(pt,(1,len(pt)))
		leafval = findcluster(pt1,n1)
		#print("keypoint:", i,":",pt1, "leaf:",leafval)
		for key in tfidf[leafval]:
			if key in visited:
				visited[key]+=1
			else:
				visited[key] = 1
		if leafval in qdic:
			qdic[leafval]+=1
		else:
			qdic[leafval] = 1

	sum1 = 0	
	for el in qdic:
		len1  = len(tfidf[el])
		qdic[el] = qdic[el]*(math.log(float(no_images)/len1)+1)		
		sum1+=(qdic[el]**2)

	sum1 = np.sqrt(sum1)	
	for el in qdic:
		qdic[el]/=sum1

	maxscore = 0
	maxim = 0
	scorelist = dict((i,0) for i in range(no_images))	
	for el in visited:
		score = 0
		d = imgdic[el]
		for key in d:
			if key in qdic:
				score+=qdic[key]*d[key]
	
		#print(score)
		scorelist[el] = score

	#print(scorelist)
	aa = filename.rsplit('/')
	aa1 = aa[len(aa)-1]
	aa2query = aa1.split('.')[0]
	clas = aa2query.split('_')[0]
	result_dir = ''.join([result_path,aa2query])
	#print("creating result folder:", result_dir)
	try:
		os.makedirs(result_dir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	ct = 0
	s = [(k, scorelist[k]) for k in sorted(scorelist, key=scorelist.get, reverse=True)]
	for im,val in s:
		aa = datatrain[im]
		aa1 = aa.rsplit('/')
		aa2 = aa1[len(aa1)-1]
		#print(aa2,val)		
		if(ct == no_result):
			break
		ct+=1
		result_file = ''.join([result_dir, "/",aa2])
		copyfile(datatrain[im], result_file)
		aa2pred = aa2.split('.')[0]
		claspred = aa2.split('_')[0]
		total+=1		
		if clas==claspred:
			correct+=1
		else:		
			print(clas, claspred, total, correct)
print("accuracy", float(correct)/total, total, correct)
#duration = 1  # second
#freq = 440  # Hz
#os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
