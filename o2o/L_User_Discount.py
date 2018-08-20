# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:46:59 2017

@author: FNo0
"""

import pandas as pd
import numpy as np

def get_L_User_Discount_Feat(dataframe_L,dataframe):
    data = dataframe_L.copy() #提特征的集
    dataset = dataframe.copy() #已构造好的集
    # 主键
    keys = list(['User_id','Discount_rate'])
    # 特征名前缀
    prefixs = 'L_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 领券次数
    pivot = pd.pivot_table(data,index = keys,values = 'Coupon_id',aggfunc = len)
    pivot.rename(columns = {'Coupon_id' : prefixs + 'Receive_Coupon_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'Receive_Coupon_cnt'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示没领券
    # 领不同类券次数
    pivot = pd.pivot_table(data,index = keys,values = 'Coupon_id',aggfunc = lambda x : len(set(x)))
    pivot.rename(columns = {'Coupon_id' : prefixs + 'Receive_different_Coupon_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'Receive_different_Coupon_cnt'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示没领券
    # 领不同商家次数
    pivot = pd.pivot_table(data,index = keys,values = 'Merchant_id',aggfunc = lambda x : len(set(x)))
    pivot.rename(columns = {'Merchant_id' : prefixs + 'Receive_different_Merchant_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'Receive_different_Merchant_cnt'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示没领券
    # 上旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x : x.day >= 1 and x.day < 11)],index = keys,values = 'Coupon_id',aggfunc = len)
    pivot.rename(columns = {'Coupon_id' : prefixs + 'Receive_Coupon_cnt_early'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'Receive_Coupon_cnt_early'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示上旬没领券
    # 中旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x : x.day >= 11 and x.day < 21)],index = keys,values = 'Coupon_id',aggfunc = len)
    pivot.rename(columns = {'Coupon_id' : prefixs + 'Receive_Coupon_cnt_middle'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'Receive_Coupon_cnt_middle'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示中旬没领券
    # 下旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x : x.day >= 21)],index = keys,values = 'Coupon_id',aggfunc = len)
    pivot.rename(columns = {'Coupon_id' : prefixs + 'Receive_Coupon_cnt_late'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'Receive_Coupon_cnt_late'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示下旬没领券
    # 最近一次领券据考察日的时间间隔
    needColumns = keys.copy()
    needColumns.append('date_received')
    testDay = np.max(data.date_received)
    timedelta = data[needColumns]
    timedelta[prefixs + 'nearestReceiveToTestDay_timedelta'] = (testDay - timedelta.date_received).map(lambda x : x.total_seconds() / (60 * 60 * 24))
    timedelta.sort_values('date_received',ascending = False,inplace = True) #按领券日期降序排列
    timedelta.drop_duplicates(keys,keep = 'first',inplace = True) #按keys去重,保留第一个
    timedelta.drop(['date_received'],axis = 1,inplace = True)
    dataset = pd.merge(dataset,timedelta,on = keys,how = 'left')
    dataset[prefixs + 'nearestReceiveToTestDay_timedelta'].fillna(-1,downcast = 'infer',inplace = True) #填充缺失值-1表示没领券
    # 最远一次领券据考察日的时间间隔
    timedelta = data[needColumns]
    timedelta[prefixs + 'furthestReceiveToTestDay_timedelta'] = (testDay - timedelta.date_received).map(lambda x : x.total_seconds() / (60 * 60 * 24))
    timedelta.sort_values('date_received',ascending = True,inplace = True) #按领券日期升序排列
    timedelta.drop_duplicates(keys,keep = 'first',inplace = True) #按keys去重,保留第一个
    timedelta.drop(['date_received'],axis = 1,inplace = True)
    dataset = pd.merge(dataset,timedelta,on = keys,how = 'left')
    dataset[prefixs + 'furthestReceiveToTestDay_timedelta'].fillna(-1,downcast = 'infer',inplace = True) #填充缺失值-1表示没领券
    # 是否第一次领券
    testDay = np.max(data.date_received)
    isFirstReceive = list(map(lambda x,y : 1 if ((testDay - x).total_seconds() / (60 * 60 * 24)) == y else 0,dataset['date_received'],dataset[prefixs + 'furthestReceiveToTestDay_timedelta']))
    dataset[prefixs + 'isFirstReceive'] = isFirstReceive
    # 是否最后一次领券
    isLastReceive = list(map(lambda x,y : 1 if ((testDay - x).total_seconds() / (60 * 60 * 24)) == y else 0,dataset['date_received'],dataset[prefixs + 'nearestReceiveToTestDay_timedelta']))
    dataset[prefixs + 'isLastReceive'] = isLastReceive
    
    return dataset
    