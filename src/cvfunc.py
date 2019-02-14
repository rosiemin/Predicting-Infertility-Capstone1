import clean_data1516 as clean
import numpy as np
import scipy.stats as scs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import merge_data1516 as md
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import StandardScaler




df, test_df = md.open_data()
X, y, df_nodum, df_full = clean.clean_data(df)


def k_fold_CV(X, y, desc, n_folds = 5, cw = 'balanced', threshold = 0.5, scaler = None):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    auc_test = []
    accuracy_test = []
    precision_test = []
    recall_test = []
    f1_score_test = []
    auc_train = []
    accuracy_train = []
    precision_train = []
    recall_train = []
    f1_score_train = []

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        model = LogisticRegression(class_weight=cw)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        probabilities = np.where(probs >= threshold, 1, 0)

        probs_train = model.predict_proba(X_train)[:, 1]
        probabilities_train = np.where(probs_train >= threshold, 1, 0)


        auc_test.append(metrics.roc_auc_score(y_test, probs))
        accuracy_test.append(metrics.accuracy_score(y_test, probabilities))
        precision_test.append(metrics.precision_score(y_test, probabilities))
        recall_test.append(metrics.recall_score(y_test, probabilities))
        f1_score_test.append(metrics.f1_score(y_test, probabilities))

        auc_train.append(metrics.roc_auc_score(y_train, probs_train))
        accuracy_train.append(metrics.accuracy_score(y_train, probabilities_train))
        precision_train.append(metrics.precision_score(y_train, probabilities_train))
        recall_train.append(metrics.recall_score(y_train, probabilities_train))
        f1_score_train.append(metrics.f1_score(y_train, probabilities_train))

    return [np.mean(auc_test), np.mean(accuracy_test), np.mean(precision_test),np.mean(recall_test), np.mean(f1_score_test), np.mean(auc_train), np.mean(accuracy_train), np.mean(precision_train), np.mean(recall_train), np.mean(f1_score_train), desc]



col = ['auc_test', 'accuracy_test', 'precision_test', 'recall_test', 'f1_score_test', 'auc_train', 'accuracy_train', 'precision_train', 'recall_train', 'f1_score_train', 'desc']
df_metrics = pd.DataFrame(columns = col)

X_train_1 = X.drop(['eth_african_american', 'eth_asian',
       'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'piv_yes'], axis = 1)
X_train_2 = X.drop(['eth_african_american', 'eth_asian',
       'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'piv_yes'], axis = 1)
X_train_3 = X.drop(['eth_african_american', 'eth_asian',
      'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'physical_activity_some', 'physical_activity_moderate','physical_activity_vigorous', 'piv_yes'], axis = 1)
X_train_4 = X[['age', 'bmi']]
X_train_5 = X[['age']]

lst = [X, X_train_1, X_train_2, X_train_3, X_train_4, X_train_5]

dslst = ['all parameters', 'no ethnicity', 'no thyroid', 'no physical act', 'age/bmi','age']
for x, desc in zip(lst, dslst):
    df_metrics.loc[len(df_metrics)] = k_fold_CV(x, y, desc)

#trying out different thresholds
t_lst = [ 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
dslst_2 = [ '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75']
for t, desc in zip(t_lst, dslst_2):
    df_metrics.loc[len(df_metrics)] = k_fold_CV(X_train_3, y, desc, threshold = t)


b_lst = ['balanced',{0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}, {0:1, 1:6}, {0:1, 1:7}, {0:1, 1:8}, {0:1, 1:9}, {0:1, 1:10}, {0:1, 1:11}]
dslst_3 = [ 'balanced','1/2', '1/3', '1/4', '1/5', '1/6', '1/7', '1/8', '1/9', '1/10', '1/11']
for c, desc in zip(b_lst, dslst_3):
    df_metrics.loc[len(df_metrics)] = k_fold_CV(X_train_3, y, desc, cw = c)
#pro con list of why i'm optimizing the way i am
## Final model?
df_metrics.loc[len(df_metrics)] = k_fold_CV(X_train_3, y, desc = "final-1/10", cw = 'balanced', threshold = 0.40)
df_metrics.loc[len(df_metrics)] = k_fold_CV(X_train_3, y, desc = "final- balance", cw = 'balanced', threshold = 0.40)

#Confusion matrix code
#
