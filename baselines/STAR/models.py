
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F

# MODEL MODES
# RNN_BASE='RNN_BASE' # Basic RNN that only uses input items (no context used)
STAR = 'STAR' # Temporal Context as input (with RNN, refer to diagram) using h3
SITAR = 'SITAR' # Combine STAR and SIAR (refer to diagram)
class SRNNModel(nn.Module):
	
	def __init__(self,num_class=2,hidden_size=40,hour_size=24,weekday_size=7,isCuda=False, mode=None):
		super(SRNNModel, self).__init__()
		self.isCuda = isCuda
		self.num_class = num_class
		self.weekday_size=weekday_size
		self.hour_size=hour_size
		self.hidden_size = hidden_size
		self.batch_size = 1
		self.mode = mode

		if self.mode == STAR:
			# (inputSize which is size of dictionary,embedding vector size)
			# ADDED THE PADDING INDEX
			self.inputEmbedding = nn.Embedding(self.num_class, self.hidden_size, padding_idx=0)
			# 1st layer params for interval context rnn
			# use hiddenLayer contains weights and bias for temporal hidden layer (h3)
			self.LinearH3 = nn.Linear(self.hidden_size,self.hidden_size)

			# Weight and bias for interval context
			self.WIntervalX = torch.nn.Parameter(torch.randn(self.hidden_size,1),requires_grad=True)
			self.BIntervalX = torch.nn.Parameter(torch.randn(self.hidden_size),requires_grad=True)

			# 2nd layer for interval context rnn
			self.Linear2ndIntervalLayer = nn.Linear(self.hidden_size,self.hidden_size*self.hidden_size)

			# Used for interval context
			self.BIntervalContext = torch.nn.Parameter(torch.randn(self.hidden_size),requires_grad=True)
			# Weight and bias parameter for x which is the input
			self.Wx = torch.nn.Parameter(torch.randn(self.hidden_size,self.hidden_size),requires_grad=True)
			self.Bx = torch.nn.Parameter(torch.randn(self.hidden_size),requires_grad=True)

			self.fc = nn.Linear(self.hidden_size,self.num_class)

		elif self.mode == SITAR:
			# (inputSize which is size of dictionary,embedding vector size)
			self.inputEmbedding = nn.Embedding(self.num_class, self.hidden_size, padding_idx=0)
			#### INPUT CONTEXT #######
			# 1st Layer params for input context rnn
			# Context Embedding (of size (1,D*D))
			self.hourEmbedding = nn.Embedding(hour_size, self.hidden_size, padding_idx=0)
			self.weekdayEmbedding = nn.Embedding(weekday_size, self.hidden_size, padding_idx=0)

			self.LinearH2 = nn.Linear(self.hidden_size,self.hidden_size)

			# Used to generate weights for input context
			self.LinearHourWeekdayX = nn.Linear(self.hidden_size*2,self.hidden_size)

			# 2nd Layer params for input context rnn
			self.Linear2ndInputContextLayer = nn.Linear(self.hidden_size,self.hidden_size*self.hidden_size)

			self.BInputContext = torch.nn.Parameter(torch.randn(self.hidden_size),requires_grad=True)

			#### INTERVAL CONTEXT #######
			# 1st layer params for interval context rnn
			# use hiddenLayer contains weights and bias for temporal hidden layer (h3)
			self.LinearH3 = nn.Linear(self.hidden_size,self.hidden_size)

			# Weight and bias for interval context
			self.WIntervalX = torch.nn.Parameter(torch.randn(self.hidden_size,1),requires_grad=True)
			self.BIntervalX = torch.nn.Parameter(torch.randn(self.hidden_size),requires_grad=True)

			# 2nd layer for interval context rnn
			self.Linear2ndIntervalLayer = nn.Linear(self.hidden_size,self.hidden_size*self.hidden_size)

			# Used for interval context
			self.BIntervalContext = torch.nn.Parameter(torch.randn(self.hidden_size),requires_grad=True)

			self.fc = nn.Linear(self.hidden_size,self.num_class)

	def forward(self, inputX,hourX,weekdayX,intervalX,h=None,h2=None,h3=None):
		# We use sigmoid as activation function
		actFunc = F.sigmoid
		if self.mode == STAR:
			# First run so initialise hidden layer
			if h is None:
				h = self.genGradZeroTensor((len(inputX), self.hidden_size))
				h3 = self.genGradZeroTensor((len(inputX), self.hidden_size))

			xEmbedded = self.inputEmbedding(inputX)
			# NOTE: h3 has a size of (1*D) and is the output of the first layer of context rnn
			h3 = actFunc(F.linear(intervalX.unsqueeze(1),self.WIntervalX,self.BIntervalX)
						+self.LinearH3(h3))
			# NOTE: we pass h3 into 2nd layer of context rnn to generate weight matrix of size (D,D)
			context_rnn_output = actFunc(self.Linear2ndIntervalLayer(h3))
			intervalContext = context_rnn_output.reshape(-1, self.hidden_size,self.hidden_size)
			part1 = F.linear(xEmbedded, self.Wx, self.Bx)
			part2 = torch.einsum('bi,bij->bj', h, intervalContext) + self.BIntervalContext
			h = actFunc(part1+part2)

			logits = self.fc(h)

		elif self.mode == SITAR:
			# First run so initialise hidden layer
			if h is None:
				h = self.genGradZeroTensor((len(inputX), self.hidden_size))
				h2 = self.genGradZeroTensor((len(inputX), self.hidden_size))
				h3 = self.genGradZeroTensor((len(inputX), self.hidden_size))

			xEmbedded = self.inputEmbedding(inputX)
			hourEmbedded = self.hourEmbedding(hourX)
			weekdayEmbedded = self.weekdayEmbedding(weekdayX)

			# cat along dim 1
			tmp1 = torch.cat((hourEmbedded,weekdayEmbedded), dim=1)
			tmp1 = self.LinearHourWeekdayX(tmp1)
			h2 = actFunc(tmp1+self.LinearH2(h2))

			inputContext = actFunc(self.Linear2ndInputContextLayer(h2))
			inputContext = inputContext.reshape(-1, self.hidden_size,self.hidden_size)

			# NOTE: h3 has a size of (1*D) and is the output of the first layer of context rnn
			tmp2 = F.linear(intervalX.unsqueeze(1), self.WIntervalX, self.BIntervalX)
			h3 = actFunc(tmp2+self.LinearH3(h3))
			# NOTE: we pass h3 into 2nd layer of context rnn to generate weight matrix of size (D,D)
			intervalContext = actFunc(self.Linear2ndIntervalLayer(h3))
			intervalContext = intervalContext.reshape(-1, self.hidden_size,self.hidden_size)

			# act(x*WxT+Bx + h*WhT+Bh)
			part1 = torch.einsum('bi,bij->bj', xEmbedded, inputContext) + self.BIntervalContext
			part2 = torch.einsum('bi,bij->bj', h, intervalContext) + self.BIntervalContext
			h = actFunc(part1+part2)

			logits =  self.fc(h)
		return logits,h,h2,h3

	def genGradZeroTensor(self,size,tensorType=None,requires_grad=True):
		# isCuda = usingCuda()
		if(tensorType==None and self.isCuda):
			# return torch.cuda.FloatTensor(size,requires_grad = requires_grad).fill_(0)
			# return torch.cuda.FloatTensor(size).fill_(0)
			return torch.zeros(size,requires_grad=requires_grad).cuda()
		elif tensorType == None:
			return torch.zeros(size,requires_grad=requires_grad)
			# return torch.zeros(size)

