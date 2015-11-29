import csv
import operator
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor


########### Test data #########################
with open('test.csv', 'rb') as f:
    reader = csv.reader(f)
    test = list(reader)
##remove header    
test.pop(0)
###############################################


########### Train data ########################
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    train = list(reader)
##remove header    
train.pop(0)
###############################################


########### Store data ########################
with open('store.csv', 'rb') as f:
    reader = csv.reader(f)
    store = list(reader)    
##remove header    
store.pop(0)

##pre-preprocess avergaes for each store
store_sum = [[0.0]*7 for i in range(1115)]
store_open = [[0]*7 for i in range(1115)]

for s in train:
    store_sum[int(s[0])-1][int(s[1])-1] += int(s[3])
    store_open[int(s[0])-1][int(s[1])-1] += int(s[5])
###############################################

    
###########defining feature matrix for a input data################
def feature(s, isTrain):
    if isTrain:
        open_days = store_open[int(s[0])-1][int(s[1])-1]
        earnings = store_sum[int(s[0])-1][int(s[1])-1]
        if (open_days == 0):
            store_rate = 0
        else:    
            store_rate = int(math.floor(earnings/open_days))
    else:
        open_days = store_open[int(s[1])-1][int(s[2])-1]
        earnings = store_sum[int(s[1])-1][int(s[2])-1]
        if (open_days == 0):
            store_rate = 0
        else:    
            store_rate = int(math.floor(earnings/open_days))
    feat = [1, store_rate]
    return feat
###################################################################


################ Train ######################
X = np.empty((0,2), int)
Y = np.empty((0,1), int)
for x in train:
    ##for training consider only values where store is open.
    if (x[5] == '1'): 
        X = np.append(X, np.array([feature(x, True)]), axis=0)
        Y = np.append(Y, np.array([[x[3]]]), axis=0)
    
rf = RandomForestRegressor(n_estimators=50, n_jobs=-1)
rf.fit(X, Y)    
###############################################


###### Write predictions ##########################    
predictions = open("submission.txt", 'w')
predictions.write("\"Id\",\"Sales\"\n")

for s in test:
    if s[4] == '0':
        predictions.write(s[0] + ",0\n")
    else:
        feat = feature(s, False)
        store_rate = np.amax(rf.predict(feat))
        predictions.write(s[0] + "," + str(store_rate) + "\n")

predictions.close()
###############################################
