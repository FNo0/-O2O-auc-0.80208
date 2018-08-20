# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 02:15:55 2017

@author: FNo0
"""

import pandas as pd

def C_User_Merchant_Feat(dataframe_C,dataframe):
    data = dataframe_C.copy() #提特征的集
    dataset = dataframe.copy() #已构造好的集
    # 主键
    keys = list(['User_id','Merchant_id'])
    # 特征名前缀
    prefixs = 'C_'
    for key in keys:
        prefixs = prefixs + key + '_'
    # 消费数
    pivot = pd.pivot_table(data,index = keys,values = 'Coupon_id',aggfunc = len)
    pivot.rename(columns = {'Coupon_id' : prefixs + 'Consume_Coupon_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'Consume_Coupon_cnt'].fillna(0,downcast = 'infer',inplace = True) #填缺失值0表示没领券
    # 领券消费数
    pivot = pd.pivot_table(data[data.Date_received != 'null'],index = keys,values = 'Coupon_id',aggfunc = len)
    pivot.rename(columns = {'Coupon_id' : prefixs + 'ReceiveAndConsume_Coupon_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'ReceiveAndConsume_Coupon_cnt'].fillna(0,downcast = 'infer',inplace = True) #填充缺失值
    # 未领券消费数
    pivot = pd.pivot_table(data[data.Date_received == 'null'],index = keys,values = 'Coupon_id',aggfunc = len)
    pivot.rename(columns = {'Coupon_id' : prefixs + 'NotReceiveConsume_Coupon_cnt'},inplace = True)
    dataset = pd.merge(dataset,pivot,left_on = keys,right_index = True,how = 'left')
    dataset[prefixs + 'NotReceiveConsume_Coupon_cnt'].fillna(0,downcast = 'infer',inplace = True) #填充缺失值
    # 领券消费数 / 消费数
    dataset[prefixs + 'ReceiveAndConsume_Coupon_cnt_div_Consume_Coupon_cnt'] = dataset[prefixs + 'ReceiveAndConsume_Coupon_cnt'] / (dataset[prefixs + 'Consume_Coupon_cnt'] + 0.1)
    # 未领券消费数 / 消费数
    dataset[prefixs + 'NotReceiveAndConsume_Coupon_cnt_div_Consume_Coupon_cnt'] = dataset[prefixs + 'NotReceiveConsume_Coupon_cnt'] / (dataset[prefixs + 'Consume_Coupon_cnt'] + 0.1)
    
    return dataset
    
    