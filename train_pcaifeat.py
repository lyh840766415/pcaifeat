import numpy as np
from loading_input import *
import random

TRAIN_FILE = 'generate_queries/pcai_training.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
BATCH_NUM_QUERIES = 2
EPOCH = 1
POSITIVES_PER_QUERY = 2
NEGATIVES_PER_QUERY = 18


#module that used to init network
#input: 
	#image_placeholder
	#pointcloud_placeholder
	
#output
	#image feature
	#pointcloud feature
	#combine feature
	#losses
	#training_ops
	
def init_pcainetwork():
	return

#module that used to load data from Hard Disk
#input
	#data information in the Hard Disk
	
#output
	#numpy matrix in the memory
def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT):
	query_pc,success_1_pc=load_pc_file(dict_value["query_pc"]) #Nx3
	query_img,success_1_img = load_image(dict_value["query_img"])

	random.shuffle(dict_value["positives"])
	pos_pc_files=[]
	pos_img_files=[]
	#load positive pointcloud
	for i in range(num_pos):
		pos_pc_files.append(QUERY_DICT[dict_value["positives"][i]]["query_pc"])
		pos_img_files.append(QUERY_DICT[dict_value["positives"][i]]["query_img"])
		
	#positives= load_pc_files(dict_value["positives"][0:num_pos])
	positives_pc,success_2_pc=load_pc_files(pos_pc_files)
	positives_img,success_2_img=load_images(pos_img_files)
	

	neg_pc_files=[]
	neg_img_files=[]
	neg_indices=[]
	random.shuffle(dict_value["negatives"])	
	for i in range(num_neg):
		neg_pc_files.append(QUERY_DICT[dict_value["negatives"][i]]["query_pc"])
		neg_img_files.append(QUERY_DICT[dict_value["negatives"][i]]["query_img"])
		neg_indices.append(dict_value["negatives"][i])
	
	negatives_pc,success_3_pc=load_pc_files(neg_pc_files)
	negatives_img,success_3_img=load_images(neg_img_files)
	if(success_1_pc and success_1_img and success_2_pc and success_2_img and success_3_pc and success_3_img):
		return [query_pc,query_img,positives_pc,positives_img,negatives_pc,negatives_img],True
	
	return [query_pc,query_img,positives_pc,positives_img,negatives_pc,negatives_img],False

#module that pass the batch_data to tensorflow placeholder
def training_one_batch():
	return

#module that log the training result and evaluate the performance
def evalute_and_log():
	return



def main():
	ops = init_pcainetwork()
	print(TRAINING_QUERIES[0])
	error_cnt = 0
	
	
	for ep in range(EPOCH):
		train_file_idxs = np.arange(0,len(TRAINING_QUERIES.keys()))
		#print(train_file_idxs)
		np.random.shuffle(train_file_idxs)
		#print(train_file_idxs)
		print('train_file_num = %f , BATCH_NUM_QUERIES = %f , iteration per batch = %f' %(len(train_file_idxs), BATCH_NUM_QUERIES,len(train_file_idxs)//BATCH_NUM_QUERIES))
		
		for i in range(len(train_file_idxs)//BATCH_NUM_QUERIES):
			batch_keys= train_file_idxs[i*BATCH_NUM_QUERIES:(i+1)*BATCH_NUM_QUERIES]
			#used to filter error data
			faulty_tuple = False
			#used to save training data
			q_tuples = []
			for j in range(BATCH_NUM_QUERIES):
				#determine whether positive & negative is enough
				if len(TRAINING_QUERIES[batch_keys[j]]["negatives"]) < NEGATIVES_PER_QUERY:
					print("Error Negative is not enough")
					faulty_tuple = True
					break
				if len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < POSITIVES_PER_QUERY:
					print("Error Positive is not enough")
					faulty_tuple = True
					break
					
				
				cur_tuples,success= get_query_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES)
				if not success:
					faulty_tuple = True
					break
					 
				q_tuples.append(cur_tuples)
				#print(q_tuples[j])
			if faulty_tuple:
				error_cnt += 1;
				continue;
	
	print("error_cnt = %d"%(error_cnt))
				
			#training_one_batch()
			
			#evaluate_and_log()

if __name__ == '__main__':
	main()