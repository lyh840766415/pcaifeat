import pickle
import numpy as np
import os
import cv2

BASE_PATH = "/"

def get_queries_dict(filename):
	#key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("Queries Loaded.")
		return queries
		
def load_pc_file(filename): 
	if not os.path.exists(os.path.join(BASE_PATH,filename)): 
		return np.array([]),False
	#returns Nx3 matrix
	pc=np.fromfile(os.path.join(BASE_PATH,filename), dtype=np.float64)

	if(pc.shape[0]!= 4096*3):
		print("pointcloud shape %d"%(pc.shape[0]//3))
		#return np.array([])

	#pc=np.reshape(pc,(pc.shape[0]//3,3))
	return pc,True

def load_pc_files(filenames):
	pcs=[]
	for filename in filenames:
		#print(filename)
		pc,success=load_pc_file(filename)
		if not success:
			return np.array([]),False
		#if(pc.shape[0]!=4096):
		#	continue
		pcs.append(pc)
	pcs=np.array(pcs)
	return pcs,True
	
def load_image(filename):
	#return scaled image
	if not os.path.exists(filename): 
		return np.array([]),False
	img = cv2.imread(filename)
	img = cv2.resize(img,(288,144))
	return img,True
	
def load_images(filenames):
	imgs=[]
	for filename in filenames:
		#print(filename)
		img,success=load_image(filename)
		if not success:
			return np.array([]),False
		imgs.append(img)
	imgs=np.array(imgs)
	return imgs,True

	