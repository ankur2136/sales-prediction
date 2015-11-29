import csv
import operator
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
### Network visualization ###
import networkx as nx
import matplotlib.pyplot as plt


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
days_count = {1:0,2:0,3:0,4:0,5:0,6:0,7:0}
days_open_count = {1:0,2:0,3:0,4:0,5:0,6:0,7:0}
customer_count = [0]*7
sale_on_day = {}

for s in train:
    store_sum[int(s[0])-1][int(s[1])-1] += int(s[3])
    store_open[int(s[0])-1][int(s[1])-1] += int(s[5])
    days_count[int(s[1])] += 1
    customer_count[int(s[1])-1] += int(s[4])
    
    if s[5] == '1':
        days_open_count[int(s[1])] += 1
    
    if (not sale_on_day.has_key(s[2])):
        sale_on_day[s[2]] = 0
    else:    
        sale_on_day[s[2]] += int(s[3])
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

fig = plt.figure(1)
ax = fig.add_subplot(111)
weekly_sum = np.sum(store_sum, axis=0)
weekly_average = np.divide(weekly_sum, days_count.values())
weekly_average_normalized = np.divide(weekly_sum, days_open_count.values())

customer_count_average = np.divide(customer_count, days_count.values())
customer_count_normalized = np.divide(customer_count, days_open_count.values())

labels = ['Monday','Tuesday','Wednesday','Thrusday','Friday','Saturday','Sunday']
dayOfWeekOfCall = [1,2,3,4,5,6,7]
plt.xticks(dayOfWeekOfCall, labels)

line1, = plt.plot(dayOfWeekOfCall, weekly_average, marker='o', label='Weekly sales average', linestyle='--')
line2, = plt.plot(dayOfWeekOfCall, weekly_average_normalized, marker='o', label='Normalized weekly sales average', linewidth=4)
line3, = plt.plot(dayOfWeekOfCall, customer_count_average, marker='o', label='Average Customer Count', linewidth=2)
line4, = plt.plot(dayOfWeekOfCall, customer_count_normalized, marker='o', label='Normalized Customer Count', linewidth=2)

##plt.legend([line1, line2, line3, line4], loc=2, borderaxespad=0.)

#plt.show()

fig2 = plt.figure(2)
x_map = range (0, len(sale_on_day.keys()))
x_axis = sale_on_day.keys()
y_axis = sale_on_day.values()
plt.xticks(x_map, x_axis)
plt.plot(x_map, y_axis)
plt.show()

