# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:47:33 2017

@author: FNo0
"""

import numpy as np
import datetime 

def preprocess(dataframe):
    data = dataframe.copy()
    data['date_received'] = data[data.Date_received != 'null'].Date_received.map(lambda x : datetime.datetime(int(str(x)[0:4]),int(str(x)[4:6]),int(str(x)[6:8]))) #转为时间格式
    if 'Date' in data.columns:
        data['date'] = data[data.Date != 'null'].Date.map(lambda x : datetime.datetime(int(str(x)[0:4]),int(str(x)[4:6]),int(str(x)[6:8]))) #转为时间格式
    return data   

def getLabel(dataframe):
    labelData = dataframe.copy()
    labelData = labelData[labelData.Date_received != 'null'] #过滤掉领券日期为空的部分
    labelData.index = range(len(labelData)) #重置index
    label = list(map(lambda date,date_received:(date - date_received).days <= 15,labelData.date,labelData.date_received)) #标签为True、False
    label = list(map(int,label)) #True、False标签转为1、0
    labelData['label'] = label #标签列
    labelData.drop(['Date','date'],axis = 1,inplace = True)
    return labelData

def splitData_By_Date_Received(dataframe,start,end):
    data = dataframe.copy()
    data = data[(data.Date_received >= start) & (data.Date_received <= end)]
    data.index = range(len(data))
    return data    

def splitData_By_Date_Consume(dataframe,start,end):
    data = dataframe.copy()
    data = data[(data.Date >= start) & (data.Date <= end)]
    data.index = range(len(data))
    return data

def dealWithReceiveDay(dataframe):
    data = dataframe.copy()
    # 周几领券
    data['weekday_Receive'] = data.date_received.map(lambda x : x.weekday())
    # 几号领券
    data['day_Receive'] = data.date_received.map(lambda x : x.day) 
    # 工作日领券
    data['isWeekdays_Receive'] = data['weekday_Receive'].map(lambda x : 1 if (x == 0 or x == 1 or x == 2 or x == 3 or x == 4) else (0 if (x == 5 or x == 6) else np.nan))
    # 周末领券
    data['isWeekends_Receive'] = data['weekday_Receive'].map(lambda x : 1 if (x == 5 or x == 6) else (0 if (x == 0 or x == 1 or x == 2 or x == 3 or x == 4) else np.nan))
    # 缺失值处理
    data['weekday_Receive'].fillna(-1,downcast = 'infer',inplace = True) #填充缺失值
    data['isWeekdays_Receive'].fillna(-1,downcast = 'infer',inplace = True) #填充缺失值
    data['isWeekends_Receive'].fillna(-1,downcast = 'infer',inplace = True) #填充缺失值
    return data

def dealwithDistance(dataframe):
    data = dataframe.copy()
    data['hasDistance'] = data.Distance.map(lambda x : 1 if x != 'null' else 0)
    data['distance'] = data.Distance.map(lambda x :-1 if x == 'null' else int(x))
    return data

def dealwithDiscount(dataframe):
    data = dataframe.copy()
    data['isManjian'] = data.Discount_rate.map(lambda x : -1 if str(x) == 'null' else (0 if ':' not in str(x) else 1))
    data['Manjian_minCost'] = data.Discount_rate.map(lambda x : -1 if (str(x) == 'null' or ':' not in str(x)) else int(str(x).split(':')[0]))
    data['discount_rate'] = data.Discount_rate.map(lambda x : -1 if str(x) == 'null' else (float(x) if ':' not in str(x) else ((float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))))
    return data
