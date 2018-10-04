#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 19:19:42 2018

@author: nikhilkonijeti
"""

import h2o
from h2o.automl import H2OAutoML
h2o.init(ip="localhost", port=54323)

df = h2o.import_file('train_u6lujuX_CVtuZ9i.csv')
df.head()

df1 = h2o.import_file('test_Y3wMUE5_7gLdaTN.csv')

df.describe()

df['Loan_Status'] = df['Loan_Status'].asfactor()

y = "Loan_Status"
x = ['Gender','Married','Education','ApplicantIncome',
    'LoanAmount','CoapplicantIncome','Loan_Amount_Term',
    'Credit_History','Property_Area']

aml = H2OAutoML(max_models = 30, max_runtime_secs=120, seed = 1)
aml.train(x = x, y = y, training_frame = df)

lb = aml.leaderboard
#lb.head()
lb.head(rows=lb.nrows)

preds = aml.predict(df1)

h2o.save_model(aml.leader, path = "./Loan_Pred_Model_III_shaz13")