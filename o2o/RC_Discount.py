# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 02:15:55 2017

@author: FNo0
"""

import pandas as pd
import numpy as np

def RC_Discount_Feat(dataframe_RC,dataframe):
    data = dataframe_RC.copy() #提特征的集
    dataset = dataframe.copy() #已构造好的集
    # 主键
    keys = list(['Discount_rate'])
    # 特征名前缀
    prefixs = 'RC_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 被领取数
    pivot = pd.pivot_table(data,index = keys,values = 'User_id',aggfunc = len)
    pivot.rename(columns = {'User_id' : prefixs + 'Received_User_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'Received_User_cnt'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示没领券
    # 被领取优惠券并使用的次数
    pivot = pd.pivot_table(data[data.Date != 'null'],index = keys,values = 'User_id',aggfunc = len)
    pivot.rename(columns = {'User_id' : prefixs + 'ReceiveAndConsumed_User_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'ReceiveAndConsumed_User_cnt'].fillna(0,downcast = 'infer',inplace = True) #填充缺失值
    # 被领取但未使用的次数
    pivot = pd.pivot_table(data[data.Date == 'null'],index = keys,values = 'User_id',aggfunc = len)
    pivot.rename(columns = {'User_id' : prefixs + 'ReceiveNotConsumed_User_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'ReceiveNotConsumed_User_cnt'].fillna(0,downcast = 'infer',inplace = True) #填充缺失值
    # 被领取并使用次数 / 被领取数
    dataset[prefixs + 'ReceiveAndConsumed_User_cnt_div_Received_cnt'] = dataset[prefixs + 'ReceiveAndConsumed_User_cnt'] / (dataset[prefixs + 'Received_User_cnt'] + 0.1)
    # 被核销的平均距离
    pivot = pd.pivot_table(data[data.Date != 'null'],index = keys,values = 'distance',aggfunc = np.mean)
    pivot.rename(columns = {'distance' : prefixs + 'ReceiveAndConsumed_User_mean_distance'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'ReceiveAndConsumed_User_mean_distance'].fillna(-1,downcast = 'infer',inplace = True) #填充缺失值
    # 多少不同商家的券被领取并消费
    pivot = pd.pivot_table(data[data.Date != 'null'],index = keys,values = 'Merchant_id',aggfunc = lambda x : len(set(x)))
    pivot.rename(columns = {'Merchant_id' : prefixs + 'ReceiveAndConsumed_in_different_Merchant_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'ReceiveAndConsumed_in_different_Merchant_cnt'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示没领
    # 多少不同商家的券领取
    pivot = pd.pivot_table(data,index = keys,values = 'Merchant_id',aggfunc = lambda x : len(set(x)))
    pivot.rename(columns = {'Merchant_id' : prefixs + 'Received_in_different_Merchant_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'Received_in_different_Merchant_cnt'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示没领
    # 多少不同商家的券被领取并消费 / 多少不同商家的券领取
    dataset[prefixs + 'ReceiveAndConsumed_in_different_Merchant_cnt_div_Received_in_different_Merchant_cnt'] = dataset[prefixs + 'ReceiveAndConsumed_in_different_Merchant_cnt'] / (dataset[prefixs + 'Received_in_different_Merchant_cnt'] + 0.1)
    # 被领取到被使用的平均间隔时间(-1表示未消费没有时间间隔)
    receiveConsumeTimedelta = data.date - data.date_received
    data['receiveConsumeTimedelta'] = receiveConsumeTimedelta.map(lambda x : x.total_seconds() / (60 * 60 * 24))
    pivot = pd.pivot_table(data,index = keys,values = 'receiveConsumeTimedelta',aggfunc = np.mean)
    pivot.rename(columns = {'receiveConsumeTimedelta' : prefixs + 'ReceiveAndConsume_timedelta'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'ReceiveAndConsume_timedelta'].fillna(-1,downcast = 'infer',inplace = True) #填充缺失值
    # 被领取到被使用的时间间隔小于等于15天的次数
    receiveConsumeTimedelta = data.date - data.date_received
    data['receiveConsumeTimedeltaNotMax15'] = data.receiveConsumeTimedelta.map(lambda x : 1 if x <= 15 else 0)
    pivot = pd.pivot_table(data,index = keys,values = 'receiveConsumeTimedeltaNotMax15',aggfunc = np.sum)
    pivot.rename(columns = {'receiveConsumeTimedeltaNotMax15' : prefixs + 'ReceiveAndConsume_timedeltaNotMax15_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'ReceiveAndConsume_timedeltaNotMax15_cnt'].fillna(0,downcast = 'infer',inplace = True) #填充缺失值
    # 被领取到被使用的时间间隔小于等于15天的次数/被领取并被使用次数
    dataset[prefixs + 'ReceiveAndConsume_timedeltaNotMax15_cnt_div_ReceiveAndConsume_cnt'] = dataset[prefixs + 'ReceiveAndConsume_timedeltaNotMax15_cnt'] / (dataset[prefixs + 'ReceiveAndConsumed_User_cnt'] + 0.1)
    # 被领取到被使用的时间间隔小于等于15天的次数/被领取数
    dataset[prefixs + 'ReceiveAndConsume_timedeltaNotMax15_cnt_div_Receive_cnt'] = dataset[prefixs + 'ReceiveAndConsume_timedeltaNotMax15_cnt'] / (dataset[prefixs + 'Received_User_cnt'] + 0.1)
    
    return dataset
    
    