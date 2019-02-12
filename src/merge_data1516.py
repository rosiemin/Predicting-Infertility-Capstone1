import numpy as np
import pandas as pd
import xport
import math
import matplotlib.pyplot as plt
import scipy.stats as scs
from sklearn.model_selection import train_test_split

def open_data():
    df_demo = pd.read_csv('../data/DEMO_I.csv')
    df_alc = pd.read_csv('../data/ALQ_I.csv')
    df_medcond = pd.read_csv('../data/MCQ_I.csv')
    df_phyact = pd.read_csv('../data/PAQ_I.csv')
    df_repo = pd.read_csv('../data/RHQ_I.csv')
    df_smoke = pd.read_csv('../data/SMQ_I.csv')
    df_sexed = pd.read_csv('../data/SXQ_I.csv')
    df_weight = pd.read_csv('../data/WHQ_I.csv')
    df_bodymea = pd.read_csv('../data/BMX_I.csv')


    df_tot = df_demo.merge(df_repo, left_on='seqn', right_on = 'seqn', how = 'left')
    df_tot = df_tot.merge(df_alc, left_on='seqn', right_on = 'seqn', how = 'left')
    df_tot = df_tot.merge(df_medcond, left_on='seqn', right_on = 'seqn', how = 'left')
    df_tot = df_tot.merge(df_phyact, left_on='seqn', right_on = 'seqn', how = 'left')
    df_tot = df_tot.merge(df_smoke, left_on='seqn', right_on = 'seqn', how = 'left')
    df_tot = df_tot.merge(df_sexed, left_on='seqn', right_on = 'seqn', how = 'left')
    df_tot = df_tot.merge(df_weight, left_on='seqn', right_on = 'seqn', how = 'left')
    df_tot = df_tot.merge(df_bodymea, left_on='seqn', right_on = 'seqn', how = 'left')

    lst = ['seqn' ,'ridageyr', 'ridreth3', 'whd010', 'whd020','mcq160m','sxq260', 'sxq265', 'sxq270', 'sxq272', 'sxq753','rhq078','paq605', 'paq620', 'paq635', 'paq706','alq130','alq141u','alq120q','smq020','smq040','rhq031',
    'rhd280','rhq305','mcq240cc','mcq240f','mcq240s','rhq074',
    'rhq076','bmxwt', 'bmxht','bmxbmi', 'bmxwaist']
    df_tot = df_tot[lst]
    # print(df_tot.shape)

    # hysterectomy RHD280 (1: yes, 2: no) 557 obs
    # ovaries removed RHQ305 (1: yes, 2: no) 287 obs
    # uterine MCQ240CC (age, or missing) 32 obs
    # cervical MCQ240F (age or missing) 22 obs
    # ovarian MCQ240S (age or missing) 21 obs

    df_tot = df_tot[(df_tot['rhd280'] != 1)]
    df_tot = df_tot[(df_tot['rhq305'] != 1)]
    df_tot = df_tot[(df_tot['mcq240cc'].isna())]
    df_tot = df_tot[(df_tot['mcq240f'].isna())]
    df_tot = df_tot[(df_tot['mcq240s'].isna())]
    # print(df_tot.shape)

    df_tot = df_tot.drop(['rhd280','rhq305','mcq240cc','mcq240f','mcq240s'], axis = 1)
    # print(df_tot.shape)


    #response variable: RHQ076, RHQ074
    df_tot['fert_stat'] = np.where((df_tot['rhq076'] == 1) | (df_tot['rhq074'] == 1), "yes", "no")
    df_tot = df_tot[~df_tot['rhq076'].isna()]
    df_tot = df_tot[~df_tot['rhq074'].isna()]
    df_tot = df_tot.drop(['rhq074','rhq076'], axis = 1)

    df_train, df_test = train_test_split(df_tot, test_size = 0.20, stratify = df_tot['fert_stat'])

    return df_train, df_test
