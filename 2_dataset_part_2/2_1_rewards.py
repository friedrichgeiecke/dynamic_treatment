# -*- coding: utf-8 -*-

"""
Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING REINFORCEMENT LEARNING
by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter

This code: estimates rewards and produces final dataset

Prerequisites: stata code ran

"""


import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import random



os.chdir("C:\\Users\\ClaudioSchilter\\Dropbox\\Reinforcement_learning\\Data\\")

dataset = pd.read_csv("the_table_ecma_withClusters.csv", sep='\t')

dataset_3v=dataset[['recid','education','age','prev_earnings','allocated_trmt','earnings','clstrs9']]


#sort data by treatment allocation
treated_data=dataset_3v.loc[dataset_3v['allocated_trmt'] == 1]
control_data=dataset_3v.loc[dataset_3v['allocated_trmt'] == 0]

X_tr = treated_data.iloc[:, 1:4].values
y_tr = treated_data.iloc[:, 5].values 

X_c = control_data.iloc[:, 1:4].values
y_c = control_data.iloc[:, 5].values 

X_tr=sm.add_constant(X_tr)
X_c=sm.add_constant(X_c)

treated_data=treated_data.reset_index(drop=True)
control_data=control_data.reset_index(drop=True)


###########
#"standard" ols rewards

model_tr = sm.OLS(y_tr,X_tr)
result_tr = model_tr.fit()
ols_tr_pred=result_tr.predict(X_tr)
ols_c_count=result_tr.predict(X_c)

model_c = sm.OLS(y_c,X_c)
result_c = model_c.fit()
ols_c_pred=result_c.predict(X_c)
ols_tr_count=result_c.predict(X_tr)

treated_data['Rlr1']=ols_tr_pred-ols_tr_count
control_data['Rlr1']=ols_c_count-ols_c_pred


###########
#doubly robust cross fitted ols rewards

#cross fitting
#get 5 folds
helper_tr=list(range(len(X_tr)))
helper_c=list(range(len(X_c)))

random.seed(123)
random.shuffle(helper_tr)
random.shuffle(helper_c)

#manually done as always rounding up/down not satisfactory
helper_tr_li=[helper_tr[0:1227],helper_tr[1227:2454],helper_tr[2454:3681],helper_tr[3681:4907],helper_tr[4907:]]
helper_c_li=[helper_c[0:618],helper_c[618:(618*2)],helper_c[(618*2):(618*3)],helper_c[(618*3):(618*4)],helper_c[(618*4):]]


#initialize DR-estimates in dataframes
treated_data['d_rob_ols_Xfit']=[0]*len(X_tr)
control_data['d_rob_ols_Xfit']=[0]*len(X_c)

for x in range(0,5):
    #for treated
    #divide up in I to get estimator and IC to get estimation for
    X_tr_IC=X_tr[helper_tr_li[x]]
    y_tr_IC=y_tr[helper_tr_li[x]]
    helper_tr_rest=list(set(helper_tr) - set(helper_tr_li[x]))
    X_tr_I=X_tr[helper_tr_rest]
    y_tr_I=y_tr[helper_tr_rest]
    #same for control
    X_c_IC=X_c[helper_c_li[x]]
    y_c_IC=y_c[helper_c_li[x]]
    helper_c_rest=list(set(helper_c) - set(helper_c_li[x]))
    X_c_I=X_c[helper_c_rest]
    y_c_I=y_c[helper_c_rest]   
    #for treated: fit using I sample 
    model_tr = sm.OLS(y_tr_I,X_tr_I)
    result_tr = model_tr.fit()
    #get predictions/counterfactual using IC sample
    ols_tr_pred=result_tr.predict(X_tr_IC)
    ols_c_count=result_tr.predict(X_c_IC)
    #same for control
    model_c = sm.OLS(y_c_I,X_c_I)
    result_c = model_c.fit()
    ols_c_pred=result_c.predict(X_c_IC)
    ols_tr_count=result_c.predict(X_tr_IC)
            
    #now get and store DR treatment-effect estimates (for I sample only)
    helperframe_tr=pd.DataFrame(data={'no':helper_tr_li[x], 'dr': (ols_tr_pred-ols_tr_count+3/2*(y_tr_IC-ols_tr_pred))})
    helperframe_c=pd.DataFrame(data={'no': helper_c_li[x], 'dr': (ols_c_count-ols_c_pred-3*(y_c_IC-ols_c_pred))})
    

    for y in helper_tr_li[x]:
        treated_data['d_rob_ols_Xfit'][y]+=helperframe_tr['dr'].loc[helperframe_tr['no']==y].values[0]


    for y in helper_c_li[x]:
        control_data['d_rob_ols_Xfit'][y]+=helperframe_c['dr'].loc[helperframe_c['no']==y].values[0]


dataset_3v=treated_data.append(control_data)
dataset_3v=dataset_3v.sort_values(by=['recid'])
dataset_3v=dataset_3v.reset_index(drop=True)

dataset_3v.to_csv("data_lpe2.csv")
