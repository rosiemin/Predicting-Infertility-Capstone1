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
import itertools
import mk_figs as mf
plt.style.use('ggplot')

df, test_df = md.open_data()
X, y, df_nodum, df_full = clean.clean_data(df)

#### TESTING TEST DATA FROM TRAIN TEST SPLIT 2
X_train_3 = X.drop(['eth_african_american', 'eth_asian',
      'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'physical_activity_some', 'physical_activity_moderate','physical_activity_vigorous','piv_yes'], axis = 1)

#std train data here
model_final = LogisticRegression(class_weight='balanced')
model_final.fit(X_train_3.values, y.values)

############ running it on the final holdout:

X_holdout, y_holdout, df_nodum_test, df_full_test = clean.clean_data(test_df)

X_holdout = X_holdout.drop(['eth_african_american', 'eth_asian',
      'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'physical_activity_some', 'physical_activity_moderate','physical_activity_vigorous', 'piv_yes'], axis = 1)

model_sm = sm.Logit(y_holdout, sm.add_constant(X_holdout))

print(model_sm.fit().summary())

probs_h = model_final.predict_proba(X_holdout)[:, 1]
probabilities_h = np.where(probs_h >= 0.4, 1, 0)
probs1_h = model_final.predict_proba(X_train_3)[:, 1]
probabilities1_h = np.where(probs1_h >= 0.4, 1, 0)

recall_final_holdout = metrics.recall_score(y_holdout, probabilities_h)
recall_train = metrics.recall_score(y, probabilities1_h)

fpr_h, tpr_h, thresholds_h = metrics.roc_curve(y_holdout, probs_h)
fpr1_h, tpr1_h, thresholds1_h = metrics.roc_curve(y, probs1_h)

auc_final_holdout = metrics.roc_auc_score(y_holdout, probs_h)
auc_train_holdout = metrics.roc_auc_score(y, probs1_h)
ROC_p = ROC_c(fpr_h, tpr_h, fpr1_h, tpr1_h, auc_final_holdout, auc_train_holdout)

conf_y_true = y_holdout.values
conf_y_true_lab = np.where(conf_y_true == 0, 'fertile', 'infertile')
conf_y_pro = probabilities_h
conf_y_pro_lab = np.where(conf_y_pro == 0, 'fertile', 'infertile')

#make all figures
cm_p = mf.generate_confusion_matrix(conf_y_true_lab, conf_y_pro_lab, labels = ['infertile', 'fertile'], title = "Confusion Matrix of Final Holdout Data", filename = '../images/confusion_mat_holdout.png', show = False)
bar_p = mf.prop_bar(df_nodum)
pair_p = mf.pair_sns(df_nodum)
box_p = mf.box_cat_plot(df_nodum)
