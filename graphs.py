import csv
import operator
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime
from scipy.interpolate import spline
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

sale_on_day = [0.0]*942
store_open_this_day = [0]*942

store_sales = [0.0]*1115
store_open_count = [0.0]*1115
store_customers = [0.0]*1115

store_262_sales = [0.0]*942
day_of_week = [0]*942

epoch = datetime.datetime.strptime("2013-01-01", "%Y-%m-%d" )
for s in train:
    store_sum[int(s[0])-1][int(s[1])-1] += int(s[3])
    store_open[int(s[0])-1][int(s[1])-1] += int(s[5])
    days_count[int(s[1])] += 1
    customer_count[int(s[1])-1] += int(s[4])
    date = datetime.datetime.strptime(s[2], "%Y-%m-%d" )
    num_days = (date - epoch).days
    day_of_week[num_days] = int(s[1])
    
    if s[5] == '1':
        days_open_count[int(s[1])] += 1
        store_sales[int(s[0])-1] += int(s[3])
        store_open_count[int(s[0])-1] += 1
        store_customers[int(s[0])-1] += int(s[4])
        
        sale_on_day[num_days] += int(s[3])
        store_open_this_day[num_days] += 1
    
    if s[0] == '262':
        store_262_sales[num_days] = int(s[3])
        
###############################################

plt.figure(1)
plt.hist(np.divide(store_sales, store_open_count), bins=100)
plt.ylabel('frequency')
plt.xlabel('Mean Sale when store was not closed')

fig = plt.figure(2)
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

plt.legend([line1, line2, line3, line4], loc=2)

fig2 = plt.figure(3)

x_sm = np.array(range (0, len(sale_on_day)))
y_sm = np.divide(sale_on_day,store_open_this_day)

x_smooth = np.linspace(x_sm.min(), x_sm.max(), 200)
y_smooth = spline(range (0, len(sale_on_day)), np.divide(sale_on_day,store_open_this_day), x_smooth)

plt.plot(x_smooth, y_smooth)

#plt.figure(4)
#plt.hist(np.divide(store_customers, store_open_count), bins=100)
#plt.ylabel('frequency')
#plt.xlabel('Mean customers per store when store was not closed')
#
#plt.figure(5)
#plt.hist(store_customers, bins=100)
#plt.ylabel('frequency')
#plt.xlabel('Mean customers per store when store was not closed')
#
#f, (ax1, ax2) = plt.subplots(1, 2)
#ax1.plot(store_customers, store_sales, 'ro')
#ax1.set_title('Customers vs Sales')
#
#ax2.plot(np.log(store_customers), np.log(store_sales), 'ro')
#ax2.set_title('log(Customers) vs log(Sales)')

plt.figure(6)
weekdays = []
sundays = []
weekdays_sale = []
weekend_sale = []
count = 0

for i,j in zip(day_of_week, store_262_sales):
    if i == 7:
        weekend_sale.append(j)
        sundays.append(count)
    else :
        weekdays_sale.append(j)
        weekdays.append(count)
    count += 1

line1, = plt.plot(weekdays, weekdays_sale, 'b+')
line2, = plt.plot(sundays, weekend_sale, 'ro')
plt.legend([line1, line2], loc=2)

plt.legend([line1, line2], ["Mon-Sat Sales", "Sunday Sales"])

plt.show()
