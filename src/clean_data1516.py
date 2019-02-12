import numpy as np
import pandas as pd
import xport
import math
import matplotlib.pyplot as plt
import scipy.stats as scs
import merge_data1516 as md

df = md.open_data().reset_index()

df.head()
df.info()
print(df.shape) #(1581, 31)

#Need to start making metrics:
#Alcohol metric -
# Column names:
# Index(['SEQN', 'RIDAGEYR', 'RIDRETH3', 'WHD010', 'WHD020', 'MCQ160M', 'SXQ260',
#        'SXQ265', 'SXQ270', 'SXQ272', 'SXQ753', 'RHQ078', 'PAQ605', 'PAQ610',
#        'PAD615', 'PAQ620', 'PAQ625', 'PAD630', 'PAQ635', 'PAD645', 'PAD660',
#        'PAD675', 'PAQ706', 'ALQ130', 'ALQ141U', 'SMQ020', 'SMQ040', 'RHQ031',
#        'RHQ074', 'RHQ076', 'infertil'],
#       dtype='object')
#
col_names = ['id', 'age', 'race']
# ALQ130 - Average number of alcoholic drinks per day in last 12 months
# Missing 540
# # ALQ141U - number of days per week/month/year had alcohol
# Missing 1180 - not a good metric, not going to use
# # ALQ120Q - how often drink alcohol over past 12 mos
# Missing 383 - Best metric, least amount missing, for this, I can impute those missing values as 0, or non-drinkers, the code book specifies that people have put 0 as their number of drinks, but when importing into python, it recodes 0s as NaNs

#For alcohol metric, I will use alq120q as my metric, i will code those as NaNs as 0, as they are missing because they do not drink.
df['alq120q'].fillna(0, inplace=True)
df.drop(['alq130', 'alq141u'])

#Smoking metric:
SXQ265
SXQ270
SXQ260
SXQ272
SXQ753

#Physical activity metric:
PAQ605
PAQ610
PAD615
PAQ620
PAQ625
PAD630
PAQ635
PAD645
PAD660
PAD675
PAQ706

#BMI new metric:
df['bmi'] = df[]
