# -*- coding: utf-8 -*

import os
import numpy as np
import random
import json
import pickle
import sys
import time
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
import models as srnn


USER_SIZE = None			
ITEM_SIZE = None			
HIDDEN_SIZE = 40			
LEARNING_RATE = 0.001 		
TOP = 10 # top 10 statistics nDCG@10, HIT@10
BEST_NDCG = 0
OUTPUT_PATH = None
OUTPUT_FILE = None

DATAFILE = None
MODEL_FILE = None
PREDICTIONS_FILE = None
OPTIM_FILE = None
MODEL_DIR = '../models/'
DATANAME = None

ITEM_TRAIN = {}
ITEM_TEST = {}
WEEKDAY_TRAIN = {}
WEEKDAY_TEST = {}
HOUR_TRAIN = {}
HOUR_TEST = {}
INTERVAL_TRAIN = {}
INTERVAL_TEST = {}

SPLIT = 0.8

# Default Mode Type
MODE_TYPE = srnn.STAR

# Context Sizes
WEEKDAY_SIZE = None
HOUR_SIZE = None
INTERVAL_SIZE = None

CUSTOM_LOSS = None

BPR_LOSS = 'BPR_LOSS'
BPR_LOSS_R = 'BPR_LOSS_R'

RUN_CUDA = False
LONG_STR = 'Long'
FLOAT_STR = 'Float'

# Log of indexes(used for nDCG)
LOG_OF_INDEXES = None

def pre_data():

	global ITEM_TRAIN
	global ITEM_TEST
	global SPLIT
	global DATAFILE

	all_cart = []
	data = open(DATAFILE, 'r')
	lines = data.readlines()
	for line in lines:
		line1 = json.loads(line)
		all_cart.append(line1)
	for i in range(len(all_cart)):
		item_train = []
		item_test = []
		weekday_train = []
		weekday_test = []
		hour_train = []
		hour_test = []
		interval_train = []
		interval_test = []

		behavior_list = all_cart[i]
		behavior_train = behavior_list[0:int(SPLIT*len(behavior_list))]
		behavior_test = behavior_list[int(SPLIT*len(behavior_list)):]

		for behavior in behavior_train:
			item_train.append(behavior[0])		
			weekday_train.append(behavior[1])
			hour_train.append(behavior[2])
			interval_train.append(behavior[3])
		for behavior in behavior_test:
			item_test.append(behavior[0])
			weekday_test.append(behavior[1])
			hour_test.append(behavior[2])
			interval_test.append(behavior[3])

		ITEM_TRAIN[i] = item_train
		ITEM_TEST[i] = item_test
		WEEKDAY_TRAIN[i] = weekday_train
		WEEKDAY_TEST[i] = weekday_test
		HOUR_TRAIN[i] = hour_train
		HOUR_TEST[i] = hour_test
		INTERVAL_TRAIN[i] = interval_train
		INTERVAL_TEST[i] = interval_test
		
def predict(model):
	relevant = 0.0 					# The total number of predictions
	hit = {}						# the number of hits in the nth position
	true_positives = {}				# The total number of hits in the first n positions (TP aka true positives)
	# nDCG where grade for each item is 1
	# nDCG  = DCG/IDCG where IDCG = 1/1
	# because in ideal case, item should be in first position
	nDCG = 0
	nDCG_full = 0
	hit_at_10 = 0

	numUsers = 0 # num of users
	numTestItem = 0

	predictionStr = ''
	
	for i in range(TOP):
		hit[i+1] = 0
		true_positives[i+1] = 0

	iteration = 0
	for n in ITEM_TEST.keys():
		# print('n in test: '+str(n))
		# iteration += 1
		# if iteration > 10:
		# 	break

		item_train = ITEM_TRAIN[n]
		item_test = ITEM_TEST[n]
		hour_train = HOUR_TRAIN[n]
		hour_test = HOUR_TEST[n]
		weekday_train = WEEKDAY_TRAIN[n]
		weekday_test = WEEKDAY_TEST[n]
		interval_train = INTERVAL_TRAIN[n]
		interval_test = INTERVAL_TEST[n]

		h = None
		h2 = None
		h3 = None
		logits = None

		# n represents each user cart, we increment user count
		numUsers +=1

		# Calculate the hidden layer corresponding to the state to be predicted
		for i in range(len(item_train)):
			inputX = item_train[i]
			hourX = hour_train[i]
			weekdayX = weekday_train[i]
			intervalX = interval_train[i]
			logits,h,h2,h3 = model(inputX,hourX,weekdayX,intervalX,h=h,h2=h2,h3=h3)

		# Forecast
		for j in range(len(item_test)):
			# Current info (Will be used for next prediction)
			inputX = item_test[j]
			hourX = hour_test[j]
			weekdayX = weekday_test[j]
			intervalX = interval_test[j]

			relevant += 1
			if CUSTOM_LOSS is None:
				probOfItems = F.softmax(logits,dim=0)
			# topK returns tuple in the form (sorted values,sorted by index)

			rankTuple = torch.topk(probOfItems, TOP)
			rank_index_list = rankTuple[1]

			if item_test[j] in rank_index_list:
				index = ((rank_index_list == item_test[j]).nonzero())
				index = index[0][0].item()
				# Remember index starts at 0 so +1 to get actual index
				# +1 more because of nDCG formula
				nDCG+= 1/getLog2AtK((index+1)+1)
				if index+1 < TOP:
					hit_at_10 += 1
	
			rankFullTuple = torch.topk(probOfItems,ITEM_SIZE)
			indexList = rankFullTuple[1]
			matchPosition = ((indexList == item_test[j]).nonzero())
			matchPosition = matchPosition[0][0].item()
			# Remember index starts at 0 so +1 to get actual index
			# +1 more because of nDCG formula
			nDCG_full+=1/getLog2AtK((matchPosition+1)+1)

			logits,h,h2,h3 = model(inputX,hourX,weekdayX,intervalX,h=h,h2=h2,h3=h3)

	#average over number of queries

	nDCG = nDCG/relevant
	nDCG_full = nDCG_full/relevant
	hit_at_10 = hit_at_10/relevant
	
	print('hit@10' + str(hit_at_10))
	print('nDCG@10: '+ str(nDCG))
	print('nDCG_full: '+str(nDCG_full))

	print('ITEM_SIZE: '+str(ITEM_SIZE))
	print('numUsers: '+str(numUsers))
	print('relevant(number of test item): '+str(relevant))


	return hit_at_10, nDCG, predictionStr
	# return true_positives

def learn():
	original_stdout = sys.stdout

	epoch = 0
		
	model = srnn.SRNNModel(hidden_size=HIDDEN_SIZE,
			weekday_size=WEEKDAY_SIZE,hour_size=HOUR_SIZE,num_class=ITEM_SIZE,isCuda=usingCuda())



	if CUSTOM_LOSS == BPR_LOSS or CUSTOM_LOSS == BPR_LOSS_R:
		criterion = torch.nn.LogSigmoid()
	else:
		criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

	if os.path.exists(MODEL_FILE):
		print("loading model to continue training")
		checkpoint = loadCheckpoint(MODEL_FILE)
		model.load_state_dict(checkpoint["model"])
		optimizer.load_state_dict(checkpoint["optim"])
		model.train()

	print("starting learning")
	print("MODEL", model)

	temp = sys.stdout
	f_handler = open(OUTPUT_FILE,'a')
	sys.stdout=f_handler
	print(model)

	f_handler.close()

	if(usingCuda()):
		model.cuda()
		print("using cuda")
		# cudnn.benchmark = True

	while (epoch<=100):
		sys.stdout=temp
		print ("Epoch %d" % epoch)
		f_handler = open(OUTPUT_FILE,'a')
		sys.stdout=f_handler
		print ("Epoch %d" % epoch)
		print ("Training...")
		sumloss = 0
		iteration = 0
		for i in ITEM_TRAIN.keys():
			iteration += 1
			# if iteration > 10:
			# 	break
			user_cart = ITEM_TRAIN[i]
			hour_cart = HOUR_TRAIN[i]
			weekday_cart = WEEKDAY_TRAIN[i]
			interval_cart = INTERVAL_TRAIN[i]
			loss = 0
			h = None
			h2 = None
			h3 = None

			# PRINT
			# sys.stdout=temp
			# print("iteration", iteration, "len user cart", len(user_cart))

			# We do not need to input the last item
			optimizer.zero_grad()

			for j in range(len(user_cart)-1):
				inputX = user_cart[j]
				hourX = hour_cart[j]
				weekdayX = weekday_cart[j]
				intervalX = interval_cart[j]
				label = user_cart[j+1]
				logits,h,h2,h3 = model(inputX,hourX,weekdayX,intervalX,h=h,h2=h2,h3=h3)
				

				loss+=criterion(logits.unsqueeze(0),label.unsqueeze(0))
			loss.backward()
			optimizer.step()
			# print('loss: '+str(loss))
			sumloss += loss

		print ("begin predict, number of test items", len(ITEM_TEST))
		sys.stdout=f_handler
		print ("begin predict")
		print('sumloss: '+str(float(sumloss)))
		hit_at_10, nDCG, predictionStr = predict(model)
		if(nDCG > BEST_NDCG):
			saveCheckpoint({
				'hiddenSize':HIDDEN_SIZE,
				'DATANAME':DATANAME,
				'BEST_NDCG':BEST_NDCG,
				'epoch':epoch,
				'optim':optimizer.state_dict(),
				'model':model.state_dict()
			})

		# save_max(recallatx, 10, ite)
		f_handler.close()
		epoch += 1
	sys.stdout = original_stdout

def saveCheckpoint(state):
	torch.save(state, MODEL_FILE)

def loadCheckpoint(filename):
	return torch.load(filename)

TENSOR_FOR_EMB = {}
def getTensorForEmb(id):
	if(id not in TENSOR_FOR_EMB):
		TENSOR_FOR_EMB[id] = genTensor([id],LONG_STR)
	return TENSOR_FOR_EMB[id]


def initLogOfIndexes():
	global LOG_OF_INDEXES
	isCuda = usingCuda()
	sizeOfArr = ITEM_SIZE+2
	if isCuda:
		cuda_str = 'cuda:'+str(torch.cuda.current_device())
		LOG_OF_INDEXES = torch.log2(torch.arange(1,sizeOfArr, device=cuda_str))
	else:
		LOG_OF_INDEXES = torch.log2(torch.arange(1,sizeOfArr, dtype=torch.float64))

def getLog2AtK(k):
	global LOG_OF_INDEXES
	position=k-1
	return LOG_OF_INDEXES[position].item()

def checkNAdd(curDict,id):
	if(id in curDict):
		curDict[id]+=1
	else:
		curDict[id]=1
	return curDict

def genTensor(tensorObj,tensorType=None):
	isCuda = usingCuda()
	if tensorType == None and isCuda:
		return torch.cuda.FloatTensor(tensorObj)
	elif tensorType == None and not isCuda:
		return torch.FloatTensor(tensorObj)
	elif tensorType == LONG_STR and isCuda:
		return torch.cuda.LongTensor(tensorObj)
	elif tensorType == LONG_STR and not isCuda:
		return torch.LongTensor(tensorObj)

def usingCuda():
	return (torch.cuda.is_available() and RUN_CUDA)


def genNegItem(user_cart):
	item = random.choice(range(1,ITEM_SIZE))
	while item in user_cart:
		item = random.choice(range(1,ITEM_SIZE))
	return item

def main():
	global ITEM_TEST,ITEM_TRAIN
	pWrite('ITEM_SIZE: '+str(ITEM_SIZE))
	pWrite('USER_SIZE: '+str(USER_SIZE))
	pre_data()
	pWrite('ITEM_TRAIN.keys(): '+str(len(ITEM_TRAIN.keys())))
	for i in ITEM_TRAIN.keys():
		for k in range(len(INTERVAL_TRAIN[i])):
			INTERVAL_TRAIN[i][k]+=1
		for k in range(len(INTERVAL_TEST[i])):
			INTERVAL_TEST[i][k]+=1
		# LIST_ITEM_TRAIN[i] = ITEM_TRAIN[i]
		ITEM_TRAIN[i] = genTensor(ITEM_TRAIN[i],tensorType=LONG_STR)
		HOUR_TRAIN[i] = genTensor(HOUR_TRAIN[i],tensorType=LONG_STR)
		WEEKDAY_TRAIN[i] = genTensor(WEEKDAY_TRAIN[i],tensorType=LONG_STR)
		INTERVAL_TRAIN[i] = genTensor(INTERVAL_TRAIN[i])
		ITEM_TEST[i] = genTensor(ITEM_TEST[i],tensorType=LONG_STR)
		HOUR_TEST[i] = genTensor(HOUR_TEST[i],tensorType=LONG_STR)
		WEEKDAY_TEST[i] = genTensor(WEEKDAY_TEST[i],tensorType=LONG_STR)
		INTERVAL_TEST[i] = genTensor(INTERVAL_TEST[i])
	learn()
	print('FINISHED LEARNING!')
	pWrite('FINISHED LEARNING!')

def pWrite(inStr):
	with open(OUTPUT_FILE, 'a') as the_file:
		the_file.write(inStr+'\n')

def createFolder(folderName):
	if not os.path.exists(folderName):
		os.makedirs(folderName)

# python main.py (cuda) (file) (model) 
# $ python main.py 1 miniData STAR 

if __name__ == '__main__':

	torch.set_printoptions(threshold=5000)

	HIDDEN_SIZE = 40
	if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
		RUN_CUDA = True
	if len(sys.argv)>2 and sys.argv[2] == 'miniData':
		DATAFILE = './data/miniData.json'
		DATANAME = 'miniData'
	elif len(sys.argv)>2 and sys.argv[2] == 'movielens':
		DATAFILE = '../../data/STAR_ml-1m.txt'
		DATANAME = 'movielens'
	elif len(sys.argv)>2 and sys.argv[2] == 'Books':
		DATAFILE = '../../data/STAR_Books.txt'
		DATANAME = 'Books'
	
	OUTPUT_PATH = './output/results/'
	MODEL_DIR= './output/model/'
	DATANAME = sys.argv[2]
	USER_SIZE = 1
	itemFreq = {}
	data = open(DATAFILE, 'r')
	lines = data.readlines()
	for line in lines:
		cart = json.loads(line)
		USER_SIZE+=1
		for element in range(len(cart)):
			curItem = cart[element][0]
			if curItem not in itemFreq:
				itemFreq[curItem] = 1
			
	ITEM_SIZE = len(itemFreq)+1
	# We add one because we do not want to start id from 0
	HOUR_SIZE = 24+1 #(hours)
	WEEKDAY_SIZE = 7+1 #(day)
	# We do not want 0 interval so we add 1
	INTERVAL_SIZE = 32 +1 

	if len(sys.argv)>3 and sys.argv[3] != None:
		MODE_TYPE = sys.argv[3]
	
	if CUSTOM_LOSS is None:
		OUTPUT_FILE = OUTPUT_PATH+MODE_TYPE+'_'+DATANAME+'_'+str(HIDDEN_SIZE)+'.txt'
		MODEL_FILE = MODEL_DIR+MODE_TYPE+'_'+DATANAME+'_'+str(HIDDEN_SIZE)+'.mdl'
	createFolder(OUTPUT_PATH)
	createFolder(MODEL_DIR)

	# Clear file
	with open(OUTPUT_FILE, 'w') as the_file:
		the_file.write("")

	print('OUTPUT_FILE: '+str(OUTPUT_FILE))
	pWrite('OUTPUT_FILE: '+str(OUTPUT_FILE))
	pWrite('RUN_CUDA: '+str(RUN_CUDA))
	pWrite('DATAFILE: '+str(DATAFILE))
	pWrite('usingCuda(): '+str(usingCuda()))
	pWrite('HIDDEN_SIZE: '+str(HIDDEN_SIZE))
	pWrite('CUSTOM_LOSS: '+str(CUSTOM_LOSS))

	initLogOfIndexes()
	start = time.time()
	main()
	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
	pWrite("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

