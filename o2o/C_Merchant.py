# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 02:15:55 2017

@author: FNo0
"""

import pandas as pd

def C_Merchant_Feat(dataframe_C,dataframe):
    data = dataframe_C.copy() #提特征的集
    dataset = dataframe.copy() #已构造好的集
    # 主键
    keys = list(['Merchant_id'])
    # 特征名前缀
    prefixs = 'C_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 消费数
    pivot = pd.pivot_table(data,index = keys,values = 'User_id',aggfunc = len)
    pivot.rename(columns = {'User_id' : prefixs + 'Consumed_User_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'Consumed_User_cnt'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示没领券
    # 领券消费数
    pivot = pd.pivot_table(data[data.Date_received != 'null'],index = keys,values = 'User_id',aggfunc = len)
    pivot.rename(columns = {'User_id' : prefixs + 'ReceiveAndConsumed_User_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'ReceiveAndConsumed_User_cnt'].fillna(0,downcast = 'infer',inplace = True) #填充缺失值
    # 未领券消费数
    pivot = pd.pivot_table(data[data.Date_received == 'null'],index = keys,values = 'User_id',aggfunc = len)
    pivot.rename(columns = {'User_id' : prefixs + 'NotReceiveConsumed_User_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'NotReceiveConsumed_User_cnt'].fillna(0,downcast = 'infer',inplace = True) #填充缺失值
    # 领券消费数 / 消费数
    dataset[prefixs + 'ReceiveAndConsumed_User_cnt_div_Consumed_User_cnt'] = dataset[prefixs + 'ReceiveAndConsumed_User_cnt'] / (dataset[prefixs + 'Consumed_User_cnt'] + 0.1)
    # 未领券消费数 / 消费数
    dataset[prefixs + 'NotReceiveAndConsumed_User_cnt_div_Consumed_User_cnt'] = dataset[prefixs + 'NotReceiveConsumed_User_cnt'] / (dataset[prefixs + 'Consumed_User_cnt'] + 0.1)
    
    return dataset
    
    