# -*- coding: utf-8 -*

'''
CA-RNN Input: with only input contexts
'''

import numpy as np
import random
import json
import pickle
import sys


# total number of users and items  in data
USER_SIZE = 1904
ITEM_SIZE = 1157

# dimensionality of hidden layer
HIDDEN_SIZE = 10

LEARNING_RATE = 0.01

# penalty rate
LAMBDA = 0.001

# recall calculated at top 20
TOP = 20

H_ZERO = np.zeros((1, HIDDEN_SIZE))
X = np.random.randn(ITEM_SIZE, HIDDEN_SIZE)*0.5
W = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE)*0.5

# first and second transfer matrices for weekday and month
UWF = []
UWS = []
UMF = []
UMS = []

# for storing optimal recalls at TOP positions - initialize as zeroes
RECALL_MAX = {}
for i in range(TOP):
	RECALL_MAX[i+1] = 0

# stores the iteration at which optimal results were obtained
ITER_MAX = 0

# initialize weekday and month transfer matrices randomly
# 7 dim for weekdays, 3 for month(?)
for i in range (7):
	uw = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE)*0.5
	UWF.append(uw)
	UWS.append(uw)
for i in range (3):
	um = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE)*0.5
	UMF.append(um)
	UMS.append(um)

# path file name
DATAFILE = 'user_cart_input.json'

ITEM_TRAIN = {}
ITEM_TEST = {}
WEEKDAY_TRAIN = {}
WEEKDAY_TEST = {}
MONTH_TRAIN = {}
MONTH_TEST = {}

# train test split
SPLIT = 0.9

def sigmoid(x):
	output = 1.0/(1.0+np.exp(-x))
	return output


def pre_data():
	"""

	Read data initialization

	ITEM_TRAIN ITEM_TEST
	"""

	global ITEM_TRAIN
	global ITEM_TEST
	global SPLIT
	global DATAFILE

	# store data from file in list all_cart
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
		month_train = []
		month_test = []

		# split behaviors into train and test
		behavior_list = all_cart[i]
		behavior_train = behavior_list[0:int(SPLIT*len(behavior_list))]
		behavior_test = behavior_list[int(SPLIT*len(behavior_list)):]

		for behavior in behavior_train:
			# at index 0 in behavior the item number is stored
			item_train.append(behavior[0])
			weekday_train.append(behavior[1])
			month_train.append(behavior[2])

		for behavior in behavior_test:
			item_test.append(behavior[0])
			weekday_test.append(behavior[1])
			month_test.append(behavior[2])

		# split cart items into different segments and store in dictionaries
		ITEM_TRAIN[i] = item_train
		ITEM_TEST[i] = item_test
		WEEKDAY_TRAIN[i] = weekday_train
		WEEKDAY_TEST[i] = weekday_test
		MONTH_TRAIN[i] = month_train
		MONTH_TEST[i] = month_test


def train(user_cart, weekday_cart, month_cart):
	global U, W, X

	# derivative of h in BPR
	dhlist = []

	# Record [1,T] status hidden layer (hidden layer not including the previous state of 1)
	hiddenlist = []

	# BPTT The derivative that is passed to the first layer sigmoid(bi)*(1-sigmoid(bi))
	midlist = []

	# initialize last hidden layer
	hl = np.copy(H_ZERO)

	# Record the total update amount of u and w in each user BPTT
	sumdUW = []
	sumdUM = []
	sumdw = 0
	loss = 0
	for i in range(7):
		sumdUW.append(0)
	for i in range(3):
		sumdUM.append(0)

	# BPR
	dh1 = np.copy(H_ZERO)				# dh for the back process

	for i in range(len(user_cart)-1):
		# 对于要预测的item进行负采样
		neg = random.randint(1, ITEM_SIZE)
		while (i+1) == neg:
			neg = random.randint(1, ITEM_SIZE)

		item_pos = X[user_cart[i+1]-1, :].reshape(1, HIDDEN_SIZE)		# positive sample's vector
		item_curt = X[user_cart[i]-1, :].reshape(1, HIDDEN_SIZE)		# current input vector
		item_neg = X[neg-1, :].reshape(1, HIDDEN_SIZE)			# negative sample's vector
		month_now = month_cart[i]
		month_next = month_cart[i+1]
		weekday_now = weekday_cart[i]
		weekday_next = weekday_cart[i+1]

		uw_now = UWF[weekday_now]
		uw_next = UWS[weekday_next]
		um_now = UMF[month_now]
		um_next = UMS[month_next]

		# 计算状态t的h、dh
		b = np.dot(item_curt, (uw_now + um_now)) + np.dot(hl, W)
		h = sigmoid(b)
		xi_j = item_pos - item_neg
		xij = np.dot(np.dot(h, (uw_next + um_next)), xi_j.T)
		loss += xij
		# 若为tmp = sigmoid(-Xij) 则LEARNING_RATE和LAMBDA为负
		tmp = -(1 - sigmoid(xij))

		hiddenlist.append(h)
		mid = h * (1 - h)
		midlist.append(mid)
		dhlist.append(tmp * np.dot(item_pos - item_neg, (uw_next.T + um_next.T)))		# save the dh for each bpr step

		# 计算对于负样本的导数 并更新负样本的vector
		dneg = -tmp * np.dot(h, (uw_next + um_next)) + LAMBDA * item_neg
		X[neg-1, :] += -LEARNING_RATE * (dneg.reshape(HIDDEN_SIZE, ))
		# 计算对于正样本的导数 并更新正样本的vector
		ditem = tmp * np.dot(h, (uw_next + um_next)) + LAMBDA * item_pos
		X[user_cart[i+1]-1, :] += -LEARNING_RATE * (ditem.reshape(HIDDEN_SIZE,))
		# 计算next UW的更新量
		# dUW = tmp * np.dot((item_pos - item_neg).T, h)
		dUWS = tmp * np.dot(h.T, (item_pos - item_neg))
		UWS[weekday_next] += -LEARNING_RATE * (dUWS + LAMBDA * UWS[weekday_next])

		dUMS = tmp * np.dot(h.T, (item_pos - item_neg))
		UMS[month_next] += -LEARNING_RATE * (dUMS + LAMBDA * UMS[month_next])
		# 更新last hidden layer
		hl = h

	# BPTT
	for i in range(len(user_cart) - 1)[::-1]:
		item = X[user_cart[i] - 1, :].reshape(1, HIDDEN_SIZE)
		month_now = month_cart[i]
		weekday_now = weekday_cart[i]
		uw_now = UWF[weekday_now]
		um_now = UMF[month_now]
		hnminus2 = hiddenlist[i]
		dh = dhlist[i] + dh1

		dUW = np.dot(item.T, dh * midlist[i])
		dUM = np.dot(item.T, dh * midlist[i])

		sumdUW[weekday_now] += dUW
		sumdUM[month_now] += dUM
		sumdw += np.dot(hnminus2.T, dh * midlist[i])
		# 更新输入的样本
		dx = np.dot(dh * midlist[i], (uw_now.T + um_now.T))
		X[user_cart[i]-1, :] += -LEARNING_RATE*(dx.reshape(HIDDEN_SIZE, ) + LAMBDA * X[user_cart[i]-1, :])

		dh1 = np.dot(dh * midlist[i], W.T)

	W += -LEARNING_RATE * (sumdw + LAMBDA * W)
	for month in range(3):
		UMF[month] += -LEARNING_RATE * (sumdUM[month] + LAMBDA * UMF[month])
	for weekday in range(7):
		UWF[weekday] += -LEARNING_RATE * (sumdUW[weekday] + LAMBDA * UWF[weekday])
	return loss


def predict():
	relevant = 0.0 			# 所预测的总次数
	hit = {}				# 第n个位置所命中的个数
	recall = {}				# 前n个位置所命中的总数
	recallatx = {}			# RecallAtN/relevant

	for i in range(TOP):
		hit[i+1] = 0
		recall[i+1] = 0

	for n in ITEM_TEST.keys():
		item_train = ITEM_TRAIN[n]
		item_test = ITEM_TEST[n]
		month_train = MONTH_TRAIN[n]
		month_test = MONTH_TEST[n]
		weekday_train = WEEKDAY_TRAIN[n]
		weekday_test = WEEKDAY_TEST[n]
		hl = np.copy(H_ZERO)
		h = np.copy(H_ZERO)
		# 计算需要预测的状态对应的hidden layer
		for i in range(len(item_train)):
			month_now = month_train[i]
			weekday_now = weekday_train[i]
			umf = UMF[month_now]
			uwf = UWF[weekday_now]
			item = X[item_train[i]-1]
			b = np.dot(item, (uwf + umf)) + np.dot(hl, W)
			h = sigmoid(b)
			hl = h
		# 预测
		for j in range(len(item_test)):
			month_now = month_test[j]
			weekday_now = weekday_test[j]
			# uwb = UWF[weekday_now]
			ums = UMS[month_now]
			uws = UWS[weekday_now]

			relevant += 1
			predict_matrix = np.dot(np.dot(h, (uws+ums)), X.T)
			rank = np.argpartition(predict_matrix[0], -TOP)[-TOP:]
			rank = rank[np.argsort(predict_matrix[0][rank])]
			rank_index_list = list(reversed(list(rank)))

			if item_test[j]-1 in rank_index_list:
				index = rank_index_list.index(item_test[j]-1)
				hit[index+1] += 1

			item = X[item_test[j] - 1]
			uwf = UWF[weekday_now]
			umf = UMF[month_now]

			b = np.dot(item, (uwf+umf)) + np.dot(h, W)
			h = sigmoid(b)

	for i in range(20):
		for j in range(20-i):
			recall[20-j] += hit[i+1]
	for i in range(20):
		recallatx[i+1] = recall[i+1]/relevant

	print(relevant)
	print(recall)
	print(recallatx)
	return recall, recallatx


def basic_info():
	print("LEARNING_RATE = %f" % LEARNING_RATE)
	print("LAMBDA = %f" % LAMBDA)

def save_max(result, n, iter):
	'''
	result:list
	保存result[n]最大的result
	'''
	global  RECALL_MAX, ITER_MAX
	if(result[n] > RECALL_MAX[n]):
		RECALL_MAX = result
		ITER_MAX = iter

	print("Best Result At Iter %i" %ITER_MAX)
	print(RECALL_MAX)
	print()

def learn():
	ite = 0
	while True:
		f_handler = open('result/input001-0001.txt','a')
		sys.stdout=f_handler
		print("Iter %d" % ite)
		print("Training...")
		sumloss = 0
		for i in ITEM_TRAIN.keys():
			user_cart = ITEM_TRAIN[i]
			month_cart = MONTH_TRAIN[i]
			weekday_cart = WEEKDAY_TRAIN[i]
			loss = train(user_cart, weekday_cart, month_cart)
			sumloss += loss

		print("begin predict")
		print(sumloss)

		recall, recallatx = predict()
		save_max(recallatx, 10, ite)
		f_handler.close()
		ite += 1


def main():

	basic_info()
	pre_data()
	learn()


if __name__ == '__main__':
	main()
