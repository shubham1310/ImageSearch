import numpy as np
import cv2
import glob
import os,errno
from sklearn.cluster import KMeans
from shutil import copyfile
import pickle
import math
from pathlib import Path

data_path = '/home/nfs/shubham9/test/proj/lavisha/101categories'
data_dir = ''.join([data_path, '/*.jpg'])
list1 = os.listdir(data_path) 
data = sorted(glob.glob(data_dir))
n = len(data)
ind = np.arange(n)
ind = np.random.permutation(ind)
n = 2500
trainno = int(0.8*n)
print("Training on ", trainno, "images")
print("Testing on ", n-trainno, "images")
trainind = ind[:trainno]
testind = ind[trainno:n]
datatrain = list( data[i] for i in trainind )
datatest = list( data[i] for i in testind )

q1 = '/home/nfs/shubham9/test/proj/lavisha/datatest.npy'
q1 = Path(q1)
if q1.is_file():
	os.remove('/home/nfs/shubham9/test/proj/lavisha/datatest.npy')
q1 = '/home/nfs/shubham9/test/proj/lavisha/datatrain.npy'
q1 = Path(q1)
if q1.is_file():
	os.remove('/home/nfs/shubham9/test/proj/lavisha/datatrain.npy')

np.save('datatest',datatest)
np.save('datatrain',datatrain)
no_images = len(datatrain)
count_leaf = 0
feature_dim = 128
max_feature = 500
no_clusters = 8
no_levels = 10
		

sift = cv2.xfeatures2d.SIFT_create()
class node:
	def __init__(self):
		self.val = None
		self.leaf = -1
		self.children = []

#Hierarchial clustering of the keypoints
def clustering(X, level, n2):
	#print("Reached",n2)
	global count_leaf
	#print("level:",level)	
	if(level>no_levels or len(X)<no_clusters):
		n2.leaf = count_leaf
		print("leaf:",count_leaf)
		count_leaf+=1
		return 
	kmeans = KMeans(n_clusters=no_clusters).fit(X)
	labels = kmeans.labels_
	clusters = [[] for i in range(no_clusters)]
	n2.val = kmeans
	for i in range(len(X)):
		#print("i:",i,"sizeofX",len(X),"labelsize",len(labels))
		clusters[labels[i]].append(X[i]);
	for i in range(no_clusters):
		temp = node()
		#print("just created",temp)
		n2.children.append(temp)
		clustering(clusters[i], level+1, n2.children[i])
	
	return n2

# Determining the clusterno of a keypoint
def findcluster(pt,n2):	
	i = n2.val.predict(pt)[0]
	#print("into cl:",i)
	if(n2.children[i].leaf!=-1):
		 return n2.children[i].leaf
	else:
		return findcluster(pt,n2.children[i])
	

print("1.Extracting SIFT keypoints from all images")

numlimit = min(700,no_images )
keypoints = np.zeros((numlimit*max_feature,feature_dim))
ct = 0
no_feature = 0
im_key = [i for i in range(no_images)]
for filename in datatrain:
	print(filename)
	img = cv2.imread(filename)
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	kp, des = sift.detectAndCompute(gray,None)
	no_key = min(max_feature,len(des))
	im_key[ct] = no_key
	ct+=1	
	keypoints[no_feature:no_feature + no_key,:] = des[0:no_key,:]
	if ct==numlimit:
		break
	no_feature+=no_key
q1 = '/home/nfs/shubham9/test/proj/lavisha/keypoints.npy'
q1 = Path(q1)
if q1.is_file():
	os.remove('/home/nfs/shubham9/test/proj/lavisha/keypoints.npy')
q1 = '/home/nfs/shubham9/test/proj/lavisha/im_key.npy'
q1 = Path(q1)
if q1.is_file():
	os.remove('/home/nfs/shubham9/test/proj/lavisha/im_key.npy')

np.save('keypoints',keypoints)
np.save('im_key',im_key)
print("Keypoint extraction done!")
print("2.Building K-tree out of ", numlimit, " images")
n1 = node()
n1 = clustering(keypoints[0:no_feature], 1, n1)	

q1 = '/home/nfs/shubham9/test/proj/lavisha/ktree.pkl'
q1 = Path(q1)
if q1.is_file():
	os.remove('/home/nfs/shubham9/test/proj/lavisha/ktree.pkl')

ktree_filename = 'ktree.pkl'
ktree_pkl = open(ktree_filename, 'wb')
pickle.dump(n1, ktree_pkl)
ktree_pkl.close()
#ktree_load_pkl = open(ktree_filename, 'rb')
#n5 = pickle.load(ktree_load_pkl)

np.save('count_leaf',count_leaf)
print("3.Populating ktree via tf-idf (for all images in dataset)")
tfidf = [{} for i in range(count_leaf)]
imgdic = [{} for i in range(no_images)]
j=0
for filename in datatrain:
	print("image:",j)
	img = cv2.imread(filename)
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	kp, des = sift.detectAndCompute(gray,None)
	no_key = min(max_feature,len(des))		
	m = no_key
	for i in range(m):
		pt = des[i]
		pt1 = np.reshape(pt,(1,len(pt)))
		leafval = findcluster(pt1,n1)
		if j in tfidf[leafval]:
			tfidf[leafval][j]+=1
		else:
			tfidf[leafval][j] = 1
		if leafval in imgdic[j]:
			imgdic[j][leafval]+=1
		else:
			imgdic[j][leafval] = 1 	
	j+=1
for j in range(no_images):
	sum1 = 0
	for key in imgdic[j]:
		len1 = len(tfidf[key])
		imgdic[j][key] = imgdic[j][key]*(math.log(float(no_images)/len1)+1)
		sum1+=(imgdic[j][key]**2)
	sum1 = np.sqrt(sum1)	
	for key in imgdic[j]:
		imgdic[j][key]/=sum1

print("All tfidf scores set")
q1 = '/home/nfs/shubham9/test/proj/lavisha/tfidf.npy'
q1 = Path(q1)
if q1.is_file():
	os.remove('/home/nfs/shubham9/test/proj/lavisha/tfidf.npy')

q1 = '/home/nfs/shubham9/test/proj/lavisha/imgdic.npy'
q1 = Path(q1)
if q1.is_file():
	os.remove('/home/nfs/shubham9/test/proj/lavisha/imgdic.npy')


np.save('tfidf',tfidf)
np.save('imgdic',imgdic)
max1 = 0
min1 = 10000000000
for dd in tfidf:
	max1 = max(max1,len(dd))
	min1 = min(min1,len(dd))

print("Max length of tfidf:", max1, "Min length of tfidf:", min1)

