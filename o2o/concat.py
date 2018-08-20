# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:50:18 2017

@author: FNo0
"""

import pandas as pd

d1 = pd.read_csv(r'./dataset/train.csv')
d2 = pd.read_csv(r'./dataset/validation.csv')

d = pd.concat([d1,d2],axis = 0)

d.to_csv(r'./dataset/new.csv',index = False)