# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:57:15 2017

@author: FNo0
"""

import prepare
import pandas as pd
import L_User
import L_Merchant
import L_Coupon
import L_Discount
import L_User_Merchant
import L_User_Coupon
import L_User_Discount
import RC_User
import RC_Merchant
import RC_Discount
import RC_User_Merchant
import RC_User_Discount
import C_User
import C_Merchant
import C_User_Merchant

def loadData():
    # 读offline数据
    offline = pd.read_csv(r'../data/ccf_offline_stage1_train.csv')
    offline.User_id = offline.User_id.astype('str')
    offline.Merchant_id = offline.Merchant_id.astype('str')
    offline.Coupon_id = offline.Coupon_id.astype('str')
    offline.Date_received = offline.Date_received.astype('str')
    print('offline读取完成!\n')
    # 读test_revised数据
    testPart = pd.read_csv(r'../data/ccf_offline_stage1_test_revised.csv')
    testPart.User_id = testPart.User_id.astype('str')
    testPart.Merchant_id = testPart.Merchant_id.astype('str')
    testPart.Coupon_id = testPart.Coupon_id.astype('str')
    testPart.Date_received = testPart.Date_received.astype('str')
    print('test_revised读取完成!\n')
    # 预处理offline和testPart
    offline = prepare.preprocess(offline)
    testPart = prepare.preprocess(testPart)
    # 处理Discount_rate
    offline = prepare.dealwithDiscount(offline)
    testPart = prepare.dealwithDiscount(testPart)
    # 处理Distance
    offline = prepare.dealwithDistance(offline)
    testPart = prepare.dealwithDistance(testPart)
    # 处理领券日期
    offline = prepare.dealWithReceiveDay(offline)
    testPart = prepare.dealWithReceiveDay(testPart)
    print('offline预处理完成!\n')
    print('test_revised预处理完成!\n')
    # 打标
    labelData = prepare.getLabel(offline)
    labelData.User_id = labelData.User_id.astype('str')
    labelData.Merchant_id = labelData.Merchant_id.astype('str')
    labelData.Coupon_id = labelData.Coupon_id.astype('str')
    labelData.Date_received = labelData.Date_received.astype('str')
    print('打标完成!\n')
    return offline,testPart,labelData

if __name__ == '__main__':
    ## 导数据与预处理
    offline,testPart,labelData = loadData()
    
    ## 训练集
    train_RC = prepare.splitData_By_Date_Received(offline,'20160301','20160430')
    train_C = prepare.splitData_By_Date_Consume(offline,'20160501','20160515')
    train_L = prepare.splitData_By_Date_Received(labelData,'20160516','20160615')
    print('构造训练集:')
    train = train_L #初始训练集
    # L特征群
    print('  L特征群:')
    train = L_User.get_L_User_Feat(train_L,train)
    print('    用户特征统计完成!')
    train = L_Merchant.get_L_Merchant_Feat(train_L,train)
    print('    商户特征统计完成!')
    train = L_Coupon.get_L_Coupon_Feat(train_L,train)
    print('    优惠券特征统计完成!')
    train = L_Discount.get_L_Discount_Feat(train_L,train)
    print('    折扣率特征统计完成!')
    train = L_User_Merchant.get_L_User_Merchant_Feat(train_L,train)
    print('    用户-商户特征统计完成!')
    train = L_User_Coupon.get_L_User_Coupon_Feat(train_L,train)
    print('    用户-优惠券特征统计完成!')
    train = L_User_Discount.get_L_User_Discount_Feat(train_L,train)
    print('    用户-折扣率特征统计完成!')
    print('  L特征群完成!')
    del train_L
    # RC特征群
    print('  RC特征群:')
    train = RC_User.RC_User_Feat(train_RC,train)
    print('    用户特征统计完成!')
    train = RC_Merchant.RC_Merchant_Feat(train_RC,train)
    print('    商户特征统计完成!')
    train = RC_Discount.RC_Discount_Feat(train_RC,train)
    print('    折扣率特征统计完成!')
    train = RC_User_Merchant.RC_User_Merchant_Feat(train_RC,train)
    print('    用户-商户特征统计完成!')
    train = RC_User_Discount.RC_User_Discount_Feat(train_RC,train)
    print('    用户-折扣率特征统计完成!')
    print('  RC特征群完成!')
    del train_RC
    # C特征群
    print('  C特征群:')
    train = C_User.C_User_Feat(train_C,train)
    print('    用户特征统计完成!')
    train = C_Merchant.C_Merchant_Feat(train_C,train)
    print('    商户特征统计完成!')
    train = C_User_Merchant.C_User_Merchant_Feat(train_C,train)
    print('    用户-商户特征统计完成!')
    print('  C特征群完成!')
    del train_C
    train.rename(columns = {'label' : 'class'},inplace = True)
    train['label'] = train['class']
    del train['class']
    train.drop(['Merchant_id','Discount_rate','Distance','date_received'],axis = 1,inplace = True)
    print('训练集划分完成!')
    
    ## 验证集
    validation_RC = prepare.splitData_By_Date_Received(offline,'20160115','20160315')
    validation_C = prepare.splitData_By_Date_Consume(offline,'20160316','20160330')
    validation_L = prepare.splitData_By_Date_Received(labelData,'20160331','20160430')
    print('构造验证集:')
    validation = validation_L #初始验证集
    # L特征群
    print('  L特征群:')
    validation = L_User.get_L_User_Feat(validation_L,validation)
    print('    用户特征统计完成!')
    validation = L_Merchant.get_L_Merchant_Feat(validation_L,validation)
    print('    商户特征统计完成!')
    validation = L_Coupon.get_L_Coupon_Feat(validation_L,validation)
    print('    优惠券特征统计完成!')
    validation = L_Discount.get_L_Discount_Feat(validation_L,validation)
    print('    折扣率特征统计完成!')
    validation = L_User_Merchant.get_L_User_Merchant_Feat(validation_L,validation)
    print('    用户-商户特征统计完成!')
    validation = L_User_Coupon.get_L_User_Coupon_Feat(validation_L,validation)
    print('    用户-优惠券特征统计完成!')
    validation = L_User_Discount.get_L_User_Discount_Feat(validation_L,validation)
    print('    用户-折扣率特征统计完成!')
    print('  L特征群完成!')
    del validation_L
    # RC特征群
    print('  RC特征群:')
    validation = RC_User.RC_User_Feat(validation_RC,validation)
    print('    用户特征统计完成!')
    validation = RC_Merchant.RC_Merchant_Feat(validation_RC,validation)
    print('    商户特征统计完成!')
    validation = RC_Discount.RC_Discount_Feat(validation_RC,validation)
    print('    折扣率特征统计完成!')
    validation = RC_User_Merchant.RC_User_Merchant_Feat(validation_RC,validation)
    print('    用户-商户特征统计完成!')
    validation = RC_User_Discount.RC_User_Discount_Feat(validation_RC,validation)
    print('    用户-折扣率特征统计完成!')
    print('  RC特征群完成!')
    del validation_RC
    # C特征群
    print('  C特征群:')
    validation = C_User.C_User_Feat(validation_C,validation)
    print('    用户特征统计完成!')
    validation = C_Merchant.C_Merchant_Feat(validation_C,validation)
    print('    商户特征统计完成!')
    validation = C_User_Merchant.C_User_Merchant_Feat(validation_C,validation)
    print('    用户-商户特征统计完成!')
    print('  C特征群完成!')
    del validation_C
    validation.rename(columns = {'label' : 'class'},inplace = True)
    validation['label'] = validation['class']
    del validation['class']
    validation.drop(['Merchant_id','Discount_rate','Distance','date_received'],axis = 1,inplace = True)
    print('验证集划分完成!')
    
    ## 测试集
    test_RC = prepare.splitData_By_Date_Received(offline,'20160416','20160615')
    test_C = prepare.splitData_By_Date_Consume(offline,'20160616','20160630')
    test_L =  testPart
    print('构造测试集:')
    test = test_L #初始测试集
    # L特征群
    print('  L特征群:')
    test = L_User.get_L_User_Feat(test_L,test)
    print('    用户特征统计完成!')
    test = L_Merchant.get_L_Merchant_Feat(test_L,test)
    print('    商户特征统计完成!')
    test = L_Coupon.get_L_Coupon_Feat(test_L,test)
    print('    优惠券特征统计完成!')
    test = L_Discount.get_L_Discount_Feat(test_L,test)
    print('    折扣率特征统计完成!')
    test = L_User_Merchant.get_L_User_Merchant_Feat(test_L,test)
    print('    用户-商户特征统计完成!')
    test = L_User_Coupon.get_L_User_Coupon_Feat(test_L,test)
    print('    用户-优惠券特征统计完成!')
    test = L_User_Discount.get_L_User_Discount_Feat(test_L,test)
    print('    用户-折扣率特征统计完成!')
    print('  L特征群完成!')
    del test_L
    # RC特征群
    print('  RC特征群:')
    test = RC_User.RC_User_Feat(test_RC,test)
    print('    用户特征统计完成!')
    test = RC_Merchant.RC_Merchant_Feat(test_RC,test)
    print('    商户特征统计完成!')
    test = RC_Discount.RC_Discount_Feat(test_RC,test)
    print('    折扣率特征统计完成!')
    test = RC_User_Merchant.RC_User_Merchant_Feat(test_RC,test)
    print('    用户-商户特征统计完成!')
    test = RC_User_Discount.RC_User_Discount_Feat(test_RC,test)
    print('    用户-折扣率特征统计完成!')
    print('  RC特征群完成!')
    del test_RC
    # C特征群
    print('  C特征群:')
    test = C_User.C_User_Feat(test_C,test)
    print('    用户特征统计完成!')
    test = C_Merchant.C_Merchant_Feat(test_C,test)
    print('    商户特征统计完成!')
    test = C_User_Merchant.C_User_Merchant_Feat(test_C,test)
    print('    用户-商户特征统计完成!')
    print('  C特征群完成!')
    del test_C
    test.drop(['Merchant_id','Discount_rate','Distance','date_received'],axis = 1,inplace = True)
    print('测试集划分完成!')

    
    train.to_csv('./dataset/train.csv',index = False)
    validation.to_csv('./dataset/validation.csv',index = False)
    test.to_csv('./dataset/test.csv',index = False)
