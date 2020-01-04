#-*-coding:GBK -*-
import os
import numpy as np
from sklearn.neighbors import KDTree
import random
import pickle

#only keep the sequence that length more than threshold
#input:
	#DATA_PATH
#output
	#selected seq_name
	#selected seq_num
def load_sequence(DATA_PATH):
	seq_num = 0
	dirs = os.listdir(DATA_PATH)
	seq_dirs=np.empty(0,dtype=object)
	for cur_dir in dirs :
		if os.path.isdir(os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"image_02/data")):
			#保证序列长度
			if len(os.listdir(os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"image_02/data"))) > 300:
				print(os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"image_02/data"))
				print(len(os.listdir(os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"image_02/data"))))
				seq_num+=1	
				seq_dirs = np.append(seq_dirs,cur_dir)
	return seq_dirs,seq_num
	
def main():
	DATA_PATH = "/home/lyh/lab/dataset/KITTI_RAW"
	seq_dirs,seq_num = load_sequence(DATA_PATH)
	
	queries = {}
	
	cnt = 0
	
	for cur_dir in seq_dirs:
		print(cur_dir)
		cur_seq = os.listdir(os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"image_02/data"))
		cur_seq.sort(key=lambda x:int(x[:-4]))
		
		#load the xyz information
		
		#print(os.path.join("./data",cur_dir,"%s.csv"%(cur_dir)))
		if not os.path.exists(os.path.join("./data",cur_dir,"%s.csv"%(cur_dir))):
			print("error 1 sequence ",cur_dir)
			continue
		
		pos = np.loadtxt(os.path.join("./data",cur_dir,"%s.csv"%(cur_dir)),delimiter=",")
		
		tree = KDTree(pos[:,0:3])
		ind_nn,_ = tree.query_radius(pos[:,0:3],r=20,sort_results = True,return_distance = True)
		ind_r,_ = tree.query_radius(pos[:,0:3], r=65,sort_results = True,return_distance = True)
		print(pos)
				
		if len(cur_seq) != pos.shape[0]:
			print("error sequence ",cur_dir)
		
		
		"""
		queries={}
	  
	  for i in range(len(ind_nn)):
		query=df_centroids.iloc[i]["file"]
		positives=np.setdiff1d(ind_nn[i],[i]).tolist()
		negatives=np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
		random.shuffle(negatives)
		queries[i]={"query":query,"positives":positives,"negatives":negatives}
		"""
			
		
		
		for i,seq in enumerate(cur_seq):
			#print(cur_dir,seq[:-4],os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"velodyne_points/data","%s.bin"%(seq[:-4])),os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"image_02/data","%s.png"%(seq[:-4])))
			positives=np.setdiff1d(ind_nn[i],[i]).tolist()
			negatives=np.setdiff1d(np.arange(0,len(cur_seq)),ind_r[i]).tolist()
			random.shuffle(positives)
			random.shuffle(negatives)
			#print(positives)
			#print(negatives)
			queries[cnt]={"cur_seq":cur_dir,"cur_num":seq[:-4],"query_img":os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"image_02/data","%s.png"%(seq[:-4])),"query_pc":os.path.join(DATA_PATH,cur_dir,cur_dir[0:10],cur_dir,"velodyne_points/data","%s.bin"%(seq[:-4])),"positives":positives,"negatives":negatives}
			#print(queries[cnt])
			cnt = cnt + 1
			if cnt%100 == 0:
				print(cnt)	
	
	filename = "pcai_training.pickle"
	with open(filename, 'wb') as handle:
	    pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("Done ", filename)
			
			
				
		

if __name__ == '__main__':
	main()