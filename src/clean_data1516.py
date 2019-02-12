import numpy as np
import pandas as pd
import xport
import math
import matplotlib.pyplot as plt
import scipy.stats as scs
import merge_data1516 as md

df, test_df = md.open_data()


def clean_data(df):
    #Need to start making metrics:

    df = df.rename(columns = {'seqn':'id', 'ridageyr':'age', 'ridreth3':'eth'})
    # ALQ130 - Average number of alcoholic drinks per day in last 12 months
    # Missing 540
    # # ALQ141U - number of days per week/month/year had alcohol
    # Missing 1180 - not a good metric, not going to use
    # # ALQ120Q - how often drink alcohol over past 12 mos
    # Missing 383 - Best metric, least amount missing, for this, I can impute those missing values as 0, or non-drinkers, the code book specifies that people have put 0 as their number of drinks, but when importing into python, it recodes 0s as NaNs

    #For alcohol metric, I will use alq120q as my metric, i will code those as NaNs as 0, as they are missing because they do not drink.
    df['alcohol'] = df['alq120q'].fillna(0)
    df = df.drop(['alq130', 'alq141u', 'paq706', 'alq120q'], axis = 1)

    # mcq160m - ever told you had thyroid problem (yes, no) (98 missing)
    # sxq260 - herpes (yes, no)
    # sxq265 - genital warts (yes, no)
    # sxq270 - gonorrhea (yes, no)
    # sxq272 - chlamydia (yes, no)
    # sxq753 - HPV (yes, no)
    # rhq078 - PIV (yes, no)
    # rhq031 - regular periods (yes, no)

    #going to make a STI indicator using herpes, genital warts, gonorrhea, chlamydia, HPV

    df['thyroid'] = np.where(df['mcq160m']==1.0, 'yes', 'no')

    df['sti'] = np.where(((df['sxq260'] == 1.0) | (df['sxq265'] == 1.0) | (df['sxq270'] == 1.0) | (df['sxq272'] == 1.0) | (df['sxq753'] == 1.0)), 'yes', 'no')

    df['piv'] = df['rhq078'].replace({1.0: 'yes', 2.0:'no', 9.0: 'no'})
    df['irr_periods'] = df['rhq031'].replace({1.0: "no", 2.0: "yes"})

    df = df.drop(['mcq160m', 'sxq260', 'sxq265', 'sxq270', 'sxq272', 'sxq753', 'rhq078', 'rhq031'], axis = 1)

    # #Physical activity metric:
    # PAQ605 - vigorous work activity 1 - yes, 2 - no - no missing
    # PAQ620 - moderate work activity (yes, no) - no missing
    # PAQ635 - walk or bike (yes, no) - no missing
    #if vigorous and moderate and walk/bike - 1
    # if moderate and walk/bike or vigorous and walk/bike - 2
    # if one of the three 3
    # none, 4

    df['vigorous'] = df['paq605']
    df['moderate'] = df['paq620']
    df['bike'] = df['paq635']

    df['physical_activity'] = df['vigorous'] + df['moderate'] + df['bike']
    df['physical_activity'].replace({6.0: 'none', 5.0: 'some', 4.0: 'moderate', 3.0: 'vigorous', 12.0: 'none'}, inplace=True)

    df = df.drop(['paq605','vigorous', 'moderate', 'bike', 'paq605', 'paq620', 'paq635'], axis = 1)
    # smoking metric
    # SMQ020 - smoked at least 100 cigarettes in life time.
    # SMQ040 - do you now smoke (1 = everyday, 2 - somedays, 3 - not at all)

    df['current'] = df['smq040'].replace({1.0: 'current', 2.0: 'current', 3.0: 'never', np.nan:'never'})
    df['smoke_cur'] = df['smq020'].replace({1.0: "former", 2.0: 'never', np.nan: 'never', 9.0: 'never'})
    df['smoke'] = np.where(df['smoke_cur'] == 'never', 'never',
             (np.where(df['current'] == 'current', 'current', 'former')))
    df = df.drop(['smq020', 'smq040','current', 'smoke_cur'], axis = 1)
    #BMI new metric:
    # Reported heights and weights, considered being physiologically implausible or the result of interviewer data entry error, were coded as “missing.”

    # WHD010 - height in inches
    # WHD020 - weight in lbs
    df['whd020_kg'] = df['whd020'] * 0.45
    df['whd010_m'] = df['whd010'] * 0.025
    df['bmi_c'] = df['whd020_kg'] / (df['whd010_m'] ** 2)

    #Identifying which rows are missing and filling in with the calculated BMI if we can
    df['bmi'] = np.where(df['bmxbmi'].isna(), df['bmi_c'],df['bmxbmi'])
    #One individual who had a BMI less than 1, doesn't make sense, so I input them as the mean of the BMI variable
    df['bmi'] = np.where(df['bmi'] < 1, np.mean(df['bmi']), df['bmi'])

    df = df.drop(['whd010', 'whd020', 'whd010_m', 'whd020_kg', 'bmxwt', 'bmxht', 'bmxwaist', 'bmxbmi','bmi_c'], axis = 1)

    df['eth'].replace({1.0: 'mexican_hispanic', 2.0: 'other_hispanic', 3.0: 'white', 4.0: 'african_american', 6.0: 'asian', 7.0: 'mixed_race'}, inplace=True)

    df.head()
    df.info()
    print(df.shape) #(1581, 31)

    df_y = df['fert_stat'].values
    df_X = df.drop('fert_stat', axis = 1)
    df_X = pd.get_dummies(df_X)
    col_remove = ['thyroid_no', 'sti_no', 'piv_no', 'irr_periods_no', 'physical_activity_none', 'smoke_never', 'eth_white']
    df_X = df_X.drop(col_remove, axis = 1).values
    return df_X, df_y
