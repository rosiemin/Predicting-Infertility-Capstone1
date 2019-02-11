import numpy as np
import pandas as pd
import xport
import math
import matplotlib.pyplot as plt
import scipy.stats as scs

df_demo = pd.read_sas('DEMO_I.XPT')
df_alc = pd.read_sas('ALQ_I.XPT')
df_medcond = pd.read_sas('MCQ_I.XPT')
df_phyact = pd.read_sas('PAQ_I.XPT')
df_repo = pd.read_sas('RHQ_I.XPT')
df_smoke = pd.read_sas('SMQ_I.XPT')
df_sexed = pd.read_sas('SXQ_I.XPT')
df_weight = pd.read_sas('WHQ_I.XPT')

df_tot = df_demo.merge(df_repo, left_on='SEQN', right_on = 'SEQN', how = 'left')
df_tot = df_tot.merge(df_alc, left_on='SEQN', right_on = 'SEQN', how = 'left')
df_tot = df_tot.merge(df_medcond, left_on='SEQN', right_on = 'SEQN', how = 'left')
df_tot = df_tot.merge(df_phyact, left_on='SEQN', right_on = 'SEQN', how = 'left')
df_tot = df_tot.merge(df_smoke, left_on='SEQN', right_on = 'SEQN', how = 'left')
df_tot = df_tot.merge(df_sexed, left_on='SEQN', right_on = 'SEQN', how = 'left')
df_tot = df_tot.merge(df_weight, left_on='SEQN', right_on = 'SEQN', how = 'left')

lst = ['SEQN', 'RIDAGEYR', 'RIDRETH3', 'WHD010', 'WHD020','MCQ160M','SXQ260', 'SXQ265', 'SXQ270', 'SXQ272', 'SXQ753','RHQ078','PAQ605', 'PAQ610', 'PAD615', 'PAQ620', 'PAQ625', 'PAD630', 'PAQ635', 'PAD645', 'PAD660', 'PAD675', 'PAQ706','ALQ130', 'ALQ141U','SMQ020', 'SMQ040','RHQ031','RHD280','RHQ305','MCQ240CC','MCQ240F','MCQ240S','RHQ074','RHQ076']
df_tot = df_tot[lst]
print(df_tot.shape)

# hysterectomy RHD280 (1: yes, 2: no) 557 obs
# ovaries removed RHQ305 (1: yes, 2: no) 287 obs
# uterine MCQ240CC (age, or missing) 32 obs
# cervical MCQ240F (age or missing) 22 obs
# ovarian MCQ240S (age or missing) 21 obs

df_tot = df_tot[(df_tot['RHD280'] != 1)]
df_tot = df_tot[(df_tot['RHQ305'] != 1)]
df_tot = df_tot[(df_tot['MCQ240CC'].isna())]
df_tot = df_tot[(df_tot['MCQ240F'].isna())]
df_tot = df_tot[(df_tot['MCQ240S'].isna())]
print(df_tot.shape)

df_tot = df_tot.drop(['RHD280','RHQ305','MCQ240CC','MCQ240F','MCQ240S'], axis = 1)
print(df_tot.shape)


#response variable: RHQ076, RHQ074
df_tot['infertil'] = np.where((df_tot['RHQ076'] == 1) | (df_tot['RHQ074'] == 1), "yes", "no")
df_tot = df_tot[~df_tot['RHQ076'].isna()]
df_tot = df_tot[~df_tot['RHQ074'].isna()]
