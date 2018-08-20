# -*- coding: utf-8 -*-
"""
Created on Tue Jul 4 10:15:17 2017

@author: yuwei
"""

import pandas as pd
import xgboost as xgb

def model_xgb(file_train,file_test):
    train = pd.read_csv(file_train)
    test = pd.read_csv(file_test)
    train_y = train['label'].values
    
    train_x = train.drop(['User_id','Coupon_id','Date_received','label'],axis=1).values
    test_x = test.drop(['User_id','Coupon_id','Date_received'],axis=1).values
 
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eta': 0.03,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 5  # 2 3
              }
    # 训练
    print('开始训练!')
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=1500,evals=watchlist)
    # 预测
    print('开始预测!')
    predict = bst.predict(dtest)
    test_xy = test[['User_id','Coupon_id','Date_received']]
    test_xy['prob'] = predict
    return test_xy
    
    
if __name__ == '__main__':
    file_train = r'./dataset/new.csv'
    print('训练集读取完成!')
    file_test = r'./dataset/test.csv'
    print('测试集读取完成!')
    predict = model_xgb(file_train,file_test)
    predict.rename(columns={0:'prob'},inplace=True)
    predict.to_csv(r'./dataset/result_train+vali.csv',index=False)
        


        

        

        
        