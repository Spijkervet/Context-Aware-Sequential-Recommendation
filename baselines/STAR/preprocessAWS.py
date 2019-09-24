import time
import operator
from datetime import datetime,timezone
import json
dayDict = {
	"Monday":1,
	"Tuesday":2,
	"Wednesday":3,
	"Thursday":4,
	"Friday":5,
	"Saturday":6,
	"Sunday":7
}

inputFileList = [
			"ratings_Books.csv",
			]

OUTPUT_FILE = "books.output"

ITEM_LIMIT = 25

LIMIT = 25

originalUserFreq = {}
originalItemFreq = {}
dupChecker = {}
originalRatingCount = 0
LineCount = 0
itemFreq = {}
ratingCount = 0
gStr = ""
RATING_STR = "rating"
ITEM_STR = "item"
TIMESTAMP_STR = "timestamp"
USER_STR = 'user'
globalUserFreq = {}
fullUserCarts ={}
pStr = ""
itemIdConverter = {}
itemIdCounter = 1

# Resets global variable
def resetGlobalVariables():
	global originalUserFreq, originalItemFreq, dupChecker,\
	originalRatingCount, LineCount, itemFreq,ratingCount,gStr, \
	fullUserCarts,itemIdConverter, itemIdCounter

	originalUserFreq = {}
	originalItemFreq = {}
	dupChecker = {}
	originalRatingCount = 0
	LineCount = 0
	itemFreq = {}
	ratingCount = 0
	# gStr = ""
	fullUserCarts ={}
	itemIdConverter = {}
	itemIdCounter = 1

# Saves to output file
def saveToFile(inputFile,):
	global gStr

	with open(OUTPUT_FILE, 'w') as the_file:
		the_file.write(gStr)

def writePreprocessedData(inputFile,finalCart):
	# Clear file
	outputFile = inputFile.replace('ratings_','')
	outputFile = outputFile.replace('../../Amazon/','')
	outputFile = outputFile.replace('csv','json')
	idConverterFile = "./data/"+outputFile
	outputFile = "./data/"+outputFile
	with open(outputFile, 'w') as the_file:
		the_file.write('')
	for i in finalCart:
		with open(outputFile, 'a') as the_file:
			the_file.write(str(i)+'\n')
	with open(idConverterFile,'w') as the_file:
		json.dump(itemIdConverter,the_file)

def writeAWSData(inputFile,finalCart):
	# Clear file
	outputFile = "output.txt"
	idConverterFile = "./data/idConv_"+outputFile
	outputFile = "./data/"+outputFile
	with open(outputFile, 'w') as the_file:
		the_file.write('')
	for i in finalCart:
		with open(outputFile, 'a') as the_file:
			the_file.write(str(i)+'\n')
	with open(idConverterFile,'w') as the_file:
		json.dump(itemIdConverter,the_file)

def p(pStr):
	global gStr
	gStr+=pStr+'\n'

def addToDict(input,diction):
	if(input in diction):
		diction[input]+=1
	else:
		diction[input]=1

def isDuplicate(item,user,rating,timestamp):
	if(user in dupChecker):
		curObj = dupChecker[user]
		if(curObj[ITEM_STR]==item
		   and curObj[RATING_STR]==rating
		   and curObj[TIMESTAMP_STR]==timestamp):
			return True
		else:
			return False
	else:
		tempObj = {}
		tempObj[ITEM_STR] = item
		tempObj[RATING_STR] = rating
		tempObj[TIMESTAMP_STR] = timestamp
		dupChecker[user] = tempObj
		return False

def convertItemToID(item):
	global itemIdCounter
	if(item not in itemIdConverter):
		itemIdConverter[item] = itemIdCounter
		itemIdCounter+=1
	if(itemIdConverter[item] > len(itemIdConverter)):
		print('itemIdConverter[item]: '+str(itemIdConverter[item]))
		print('itemIdCounter: '+str(itemIdCounter))
	# assert(itemIdConverter[item] > len(itemIdConverter))
	return itemIdConverter[item]

def addToCart(item,user,timestamp):
	global fullUserCarts
	if(user not in fullUserCarts):
		fullUserCarts[user] = []

	fullUserCarts[user].append({
			USER_STR:user,
			ITEM_STR:convertItemToID(item),
			TIMESTAMP_STR:int(timestamp)
		})


def processCart(cart):
	# print(cart)
	cart = sorted(cart, key=operator.itemgetter(TIMESTAMP_STR))
	processedCart = []
	for i in range(len(cart)):
		curSet = cart[i]
		# Days context (7 days)
		# day = dayDict[date.fromtimestamp(curSet[TIMESTAMP_STR]).strftime('%A')]
		day = dayDict[(time.strftime('%A', time.localtime(curSet[TIMESTAMP_STR])))]
		# Hours context (24hr)
		hour = int(time.strftime('%H', time.localtime(curSet[TIMESTAMP_STR])))
		# Interval context (30 or greater) (32 in total)
		# Note we start interval context with 1 (do this model code, here just start with 0)
		interval = None
		if i == 0:
			interval = 0
			# checkNAdd(temporalFreqDist,0)
		else:
			prevTimeStamp = cart[i-1][TIMESTAMP_STR]
			prevDate = datetime.fromtimestamp(prevTimeStamp)
			# print('prevDate: '+str(prevDate.strftime('%Y-%m-%d %H:%M:%S')))
			curDate = datetime.fromtimestamp(curSet[TIMESTAMP_STR])
			# print('curDate: '+str(curDate.strftime('%Y-%m-%d %H:%M:%S')))
			dayDiff = (curDate - prevDate).days
			if(dayDiff > 30):
				interval = 31
			else:
				interval = dayDiff
			secDiff = curSet[TIMESTAMP_STR]-prevTimeStamp

		processedCart.append([int(curSet[ITEM_STR]),int(day),int(hour),int(interval)])

	return processedCart

def initialProcessFile(inputFile): 
	global originalItemFreq
	count = 0
	with open(inputFile, encoding="utf8") as infile:
		p('filename: '+str(inputFile))
		for line in infile:
			line = line.replace('\n','')
			item,user,rating,timestamp, _ = line.split(',')
			isDup = isDuplicate(item,user,rating,timestamp)
			if(not isDup):
				addToDict(item,originalItemFreq)
			else:
				print('user dup: '+str(user))

def processFile(inputFile):
	global LineCount,originalUserFreq,itemFreq,originalRatingCount, \
	globalUserFreq
	count = 0
	with open(inputFile, encoding="utf8") as infile:
		p('filename: '+str(inputFile))
		for line in infile:
			LineCount+=1
			line = line.replace('\n','')
			item,user,rating,timestamp, _ = line.split(',')
			isDup = isDuplicate(item,user,rating,timestamp)
			if(originalItemFreq[item] >= ITEM_LIMIT):
				addToDict(user,originalUserFreq)
				originalRatingCount+=1
				addToDict(user,globalUserFreq)
				# addToCart(item,user,timestamp)
			else:
				print('low freq item: '+str(item))
				print('originalItemFreq[item]: '+str(originalItemFreq[item]))


def reProcessFile(inputFile):
	global originalUserFreq,itemFreq,ratingCount
	count = 0
	with open(inputFile, encoding="utf8") as infile:
		p('filename: '+str(inputFile))
		for line in infile:
			line = line.replace('\n','')
			item,user,rating,timestamp, _ = line.split(',')
			if(user in originalUserFreq and 
				originalUserFreq[user]>=LIMIT):
				addToDict(item,itemFreq)
				ratingCount +=1
				addToCart(item,user,timestamp)
			else:
				print('low freq user: '+str(user))

def analyseFile(inputFile):
	resetGlobalVariables()
	print('inputFile: '+str(inputFile))

	# PROCESS FILE TO GET ITEM FREQ
	initialProcessFile(inputFile)

	# PROCESS FILE TO GET NUMBER OF RATINGS PER USER
	processFile(inputFile)
	p("Before processing")
	p('LineCount: '+str(LineCount))

	p('originalUserCount: '+str(len(originalUserFreq)))
	p('originalItemCount: '+str(len(originalItemFreq)))
	p('originalRatingCount: '+str(originalRatingCount))


	userCount = 0
	for key in originalUserFreq:
		if(originalUserFreq[key]>=LIMIT):
			userCount+=1
	# REPROCESS FILE TO EXCLUDE USERS WITH LESS THAN 10 RATINGS
	reProcessFile(inputFile)
	finalCart = []
	for i in fullUserCarts:
		if(originalUserFreq[i]>=LIMIT):
			finalCart.append(processCart(fullUserCarts[i]))

	p("After processing")
	p('userCount: '+str(userCount))
	p('itemCount: '+str(len(itemFreq)))
	p('ratingCount: '+str(ratingCount))
	# saveToFile(inputFile)
	p('')
	writePreprocessedData(inputFile,finalCart)

def addToRate(input,diction):
	if(input in diction):
		diction[input]+=1
	else:
		print('input: '+str(input))
		raise ValueError('input in diction')



def n_initProcess(inputFile,userRate,itemRate):
	totalRating = 0
	with open(inputFile, encoding="utf8") as infile:
		p('inputFile: '+str(inputFile))
		for line in infile:
			line = line.replace('\n','')
			item,user,rating,timestamp, _ = line.split(' ')
			addToDict(user,userRate)
			addToDict(item,itemRate)
			totalRating+=1
	return userRate,itemRate,totalRating

def n_removeUserNItem(userRate,itemRate):
	rmUserList = []
	rmItemList = []
	for key in userRate:
		if userRate[key] < LIMIT:
			print( userRate[key], 'gets removed')
			# del userRate[key]
			rmUserList.append(key)
	for key in itemRate:
		if itemRate[key] < ITEM_LIMIT:
			# del itemRate[key]
			rmItemList.append(key)
	for i in range(len(rmUserList)):
		key = rmUserList[i]
		del userRate[key]
	for i in range(len(rmItemList)):
		key = rmItemList[i]
		del itemRate[key]
	return userRate,itemRate
		
def n_reprocess(inputFile,userRate,itemRate):
	newUserRate = {}
	newItemRate = {}
	totalRating = 0
	with open(inputFile, encoding="utf8") as infile:
		# p('inputFile: '+str(inputFile))
		for line in infile:
			line = line.replace('\n','')
			item,user,rating,timestamp, _= line.split(' ')
			if user in userRate and item in itemRate:
				if(userRate[user] < LIMIT):
					print('user: '+str(user))
					print('userRate[user]: '+str(userRate[user]))
					raise ValueError('User lesser than limit')
				if(itemRate[item] < ITEM_LIMIT):
					print('item: '+str(item))
					print('itemRate[item]: '+str(itemRate[item]))
					raise ValueError('Item lesser than item limit')
				addToDict(user,newUserRate)
				addToDict(item,newItemRate)
				totalRating +=1
	return newUserRate,newItemRate,totalRating

def n_finalRun(inputFile,userRate,itemRate):
	userData = {}
	totalRating = 0
	with open(inputFile, encoding="utf8") as infile:
		# p('inputFile: '+str(inputFile))
		for line in infile:
			line = line.replace('\n','')
			item,user,rating,timestamp, _ = line.split(' ')
			if user in userRate and item in itemRate:
				if user not in userData:
					userData[user] = []
				totalRating+=1
				userData[user].append({
					USER_STR:user,
					ITEM_STR:convertItemToID(item),
					TIMESTAMP_STR:int(timestamp)
					})
	return userData,totalRating

def run(inputFile):
	resetGlobalVariables()	
	print('inputFile: '+str(inputFile))
	p('inputFile: '+str(inputFile))
	userRate = {}
	itemRate = {}
	totalRating = 0
	userRate,itemRate,totalRating = n_initProcess(inputFile,userRate,itemRate)

	print('original user count: '+str(len(userRate)))
	print('original user count: '+str(len(itemRate)))
	print('totalRating: '+str(totalRating))
	step = 1
	while(True):
		print('step: '+str(step))
		totalUser = len(userRate)
		totalItem = len(itemRate)
		userRate,itemRate = n_removeUserNItem(userRate,itemRate)
		if totalUser == len(userRate) and totalItem == len(itemRate):
			break
		else:
			userRate,itemRate,totalRating = n_reprocess(inputFile,userRate,itemRate)
		step+=1
	p('After processing')
	p('--------------------------')
	p('user count: '+str(len(userRate)))
	p('item count: '+str(len(itemRate)))

	userData,checkRating = n_finalRun(inputFile,userRate,itemRate)
	if(checkRating != totalRating):
		print('totalRating: '+str(totalRating))
		print('checkRating: '+str(checkRating))
		raise ValueError('checkRating dont match up with totalRating')
	totalCart = []
	for i in userData:
		totalCart.append(processCart(userData[i]))
	p('totalRating: '+str(checkRating))

	# saveToFile(inputFile)
	p('')
	writeAWSData(inputFile,totalCart)	

import time
# STARTING POINT
gStart = time.time()
for i in range(len(inputFileList)):
	# analyseFile('../../AWS/'+str(inputFileList[i]))
	start_time = time.time()
	# run('./rawData/'+str(inputFileList[i]))
	run('../../data/Books.txt')
	e = time.time() - start_time
	print('{:02d}:{:02d}:{:02d}'.format(int(e // 3600), int((e % 3600 // 60)), int(e % 60)))
	p('{:02d}:{:02d}:{:02d}'.format(int(e // 3600), int((e % 3600 // 60)), int(e % 60)))
	p("")
print('total')
e = time.time() - gStart
print('{:02d}:{:02d}:{:02d}'.format(int(e // 3600), int((e % 3600 // 60)), int(e % 60)))
p('Total Time for preprocessing')
p('{:02d}:{:02d}:{:02d}'.format(int(e // 3600), int((e % 3600 // 60)), int(e % 60)))

saveToFile("")
