import numpy as np
from loading_input import *
from pointnetvlad.pointnetvlad_cls import *
import random
import cv2
import nets.resnet_v1_50 as resnet
import tensorflow as tf


TRAIN_FILE = 'generate_queries/pcai_training.pickle'
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
BATCH_NUM_QUERIES = 2
EPOCH = 1
POSITIVES_PER_QUERY = 2
NEGATIVES_PER_QUERY = 2

def get_bn_decay(batch):
	#batch norm parameter
	DECAY_STEP = 20000
	BN_INIT_DECAY = 0.5
	BN_DECAY_DECAY_RATE = 0.5
	BN_DECAY_DECAY_STEP = float(DECAY_STEP)
	BN_DECAY_CLIP = 0.99
	bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,batch*BATCH_NUM_QUERIES,BN_DECAY_DECAY_STEP,BN_DECAY_DECAY_RATE,staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

#module that uesed to extract image feature
#input
	#image data
#output
	#image feature
def init_imgnetwork():
	images_placeholder = tf.placeholder(tf.float32,shape=[None,144,288,3])
	embbed_size = 128
	endpoints,body_prefix = resnet.endpoints(images_placeholder,is_training=True)
	#fix output feature to 128-d
	img_feat = tf.layers.dense(endpoints['model_output'], embbed_size)
	return images_placeholder,img_feat


#module that uesed to extract point cloud feature
#input
	#point cloud data
#output
	#point cloud feature
def init_pcnetwork():
	batch_size = 2
	embbed_size = 128
	pc_placeholder = tf.placeholder(tf.float32,shape=[batch_size,1,4096,3])
	is_training_pl = tf.Variable(True, name = 'is_training')
	bn_decay = tf.Variable(1.0,name = 'bn_decay')
	#bn_decay = get_bn_decay(100)
	
	print(bn_decay)
	
	endpoints = pointnetvlad(pc_placeholder,is_training_pl,bn_decay)
	pc_feat = tf.layers.dense(endpoints,embbed_size)
	return pc_placeholder,pc_feat
	
	

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
	images_placeholder,img_feat = init_imgnetwork()
	pc_placeholder,pc_feat = init_pcnetwork()
	
	
	return images_placeholder,pc_placeholder,img_feat,pc_feat

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
	images_placeholder,pc_placeholder,img_feat,pc_feat = init_pcainetwork()
	print(TRAINING_QUERIES[0])
	error_cnt = 0
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	saver = tf.train.Saver()
	
	#Start training
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
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
					
				if faulty_tuple:
					error_cnt += 1;
					continue;
				
				for que_i, cur in enumerate(q_tuples):
					query_pc = cur[0]
					query_pc = np.expand_dims(query_pc,axis = 0)
					query_img = cur[1]
					query_img = np.expand_dims(query_img,axis = 0)
					positives_pc = cur[2]
					positives_img = cur[3]
					negatives_pc = cur[4]
					negatives_img = cur[5]
					
					img = np.concatenate((query_img,positives_img,negatives_img),axis=0)
					pc = np.concatenate((query_pc,positives_pc,negatives_pc),axis=0)
					print(img.shape,pc.shape)
					
					print(query_pc.shape,query_img.shape,positives_pc.shape,positives_img.shape,negatives_pc[0].shape,negatives_img.shape)
					for j in range(img.shape[0]):
						cv2.imwrite("%d_%d.png"%(que_i,j),img[j])
						
					for j in range(pc.shape[0]):
						np.savetxt("%d_%d.txt"%(que_i,j), pc[j], fmt="%.5f", delimiter = ',')
				exit()
				
				train_feed_dict = {
					images_placeholder:cur_tuples[3]				
				}
				#image_feat = sess.run([logits],feed_dict = train_feed_dict)
				
				print(image_feat)
	
	print("error_cnt = %d"%(error_cnt))
				
			#training_one_batch()
			
			#evaluate_and_log()

if __name__ == '__main__':
	main()