#load position gps/ins data

#load pointcloud pos

#load image time and give it time

#for seq in all seqs

#for pcl in pointclouds
#determine train/test
#search positive pointcloud&img
#search negative pointcloud&img

#save to file

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree

#load position gps/ins data
POS_FILE_PATH = "/home/lyh/lab/benchmark_datasets/oxford_img/2014-05-19-13-20-57/2014-05-19-13-20-57/gps/ins.csv"
pos_data = pd.read_csv(POS_FILE_PATH)
print(pos_data.shape)
col_n = ['timestamp','northing','easting']
pos_data = pd.DataFrame(pos_data,columns = col_n)
print(pos_data)
pos_data = np.array(pos_data)
print(pos_data[:,0:1])
time_tree = KDTree(pos_data[:,0:1])
	
	
#load image time and give it time
IMG_TIME_FILE_PATH = "/home/lyh/lab/benchmark_datasets/oxford_img/2014-05-19-13-20-57/2014-05-19-13-20-57/stereo.timestamps"
image_time = np.loadtxt(IMG_TIME_FILE_PATH)
print(image_time[:,0])
	
	
	

nearest_time_dis,nearest_time_index = time_tree.query(image_time[:,0:1],k=1)
print(nearest_time_index)
print(nearest_time_index.shape)
exit()




#print(csv_data.iloc[:,0].values)


