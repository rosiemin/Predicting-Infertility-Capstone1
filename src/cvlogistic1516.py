import clean_data1516 as clean
import numpy as np
import scipy.stats as scs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import merge_data1516 as md
import statsmodels.api as sm
import seaborn as sns


df, test_df = md.open_data()
X, y, df_nodum, df_full = clean.clean_data(df)

######## EDA FIRST ##########

# examine relationships between my continuous variables and my outcomes excluding id


################## ALL Predictors ##################
#cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y)

# Running logistic regression with stats models:
model_sm = sm.Logit(y_train, sm.add_constant(X_train))

print(model_sm.fit().summary())


model = LogisticRegression(class_weight='balanced')
model.fit(X_train.values, y_train.values)
probabilities = model.predict(X_test)#
probs = model.predict_proba(X_test)[:, 1]
#
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
auc = metrics.roc_auc_score(y_test, probs)
accuracy = metrics.accuracy_score(y_test, probabilities)
precision = metrics.precision_score(y_test, probabilities)
recall = metrics.recall_score(y_test, probabilities)
f1_score = metrics.f1_score(y_test, probabilities)

col = ['auc', 'accuracy', 'precision', 'recall', 'f1_score', 'desc']
df_metrics = pd.DataFrame(columns = col)
df_metrics.loc[len(df_metrics)] = [auc, accuracy, precision, recall, f1_score, 'all pred']

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
ax.plot(fpr, tpr, color='b', lw=2, label='Model')
ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Postive Rate", fontsize=20)
ax.set_title("ROC curve: All predictors", fontsize=24)
ax.text(0.3, 0.7, " ".join(["AUC:",str(auc.round(3))]), fontsize=20)
ax.legend(fontsize=24)
plt.savefig('../images/ROC_allPred.png')

############## REMOVED ETHNICITY ################
## Based on the output, I am going to remove ethnicity, as it isn't a known risk factor of infertility.

X_train_1 = X_train.drop(['eth_african_american', 'eth_asian',
       'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic'], axis = 1)
X_test_1 = X_test.drop(['eth_african_american', 'eth_asian',
       'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic'], axis = 1)

model_sm_1 = sm.Logit(y_train, sm.add_constant(X_train_1))

print(model_sm_1.fit().summary())


model_1 = LogisticRegression(class_weight='balanced')
model_1.fit(X_train_1.values, y_train.values)
probabilities = model_1.predict(X_test_1)
probs = model_1.predict_proba(X_test_1)[:, 1]

accuracy_1 = metrics.accuracy_score(y_test, probabilities)
precision_1 = metrics.precision_score(y_test, probabilities)
recall_1 = metrics.recall_score(y_test, probabilities)
f1_score_1 = metrics.f1_score(y_test, probabilities)


fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
auc_1 = metrics.roc_auc_score(y_test, probs)

df_metrics.loc[len(df_metrics)] = [auc_1, accuracy_1, precision_1, recall_1,f1_score_1, 'no ethnicity']


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
ax.plot(fpr, tpr, color='b', lw=2, label='Model')
ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Postive Rate", fontsize=20)
ax.set_title("ROC curve: Ethnicity Removed", fontsize=24)
ax.text(0.3, 0.7, " ".join(["AUC:",str(auc_1.round(3))]), fontsize=20)
ax.legend(fontsize=24)
plt.savefig('../images/ROC_no_eth.png')

############## removing thyroid ###########

X_train_2 = X_train.drop(['eth_african_american', 'eth_asian',
       'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes'], axis = 1)
X_test_2 = X_test.drop(['eth_african_american', 'eth_asian',
       'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes'], axis = 1)

model_sm_2 = sm.Logit(y_train, sm.add_constant(X_train_2))

print(model_sm_2.fit().summary())


model_2 = LogisticRegression(class_weight='balanced')
model_2.fit(X_train_2.values, y_train.values)
probabilities = model_2.predict(X_test_2)
probs = model_2.predict_proba(X_test_2)[:, 1]

accuracy_2 = metrics.accuracy_score(y_test, probabilities)
precision_2 = metrics.precision_score(y_test, probabilities)
recall_2 = metrics.recall_score(y_test, probabilities)
f1_score_2 = metrics.f1_score(y_test, probabilities)


fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
auc_2 = metrics.roc_auc_score(y_test, probs)

df_metrics.loc[len(df_metrics)] = [auc_2, accuracy_2, precision_2, recall_2,f1_score_2, 'no thyroid']


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
ax.plot(fpr, tpr, color='b', lw=2, label='Model')
ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Postive Rate", fontsize=20)
ax.set_title("ROC curve: Ethnicity/Thyroid Removed", fontsize=24)
ax.text(0.3, 0.7, " ".join(["AUC:",str(auc_2.round(3))]), fontsize=20)
ax.legend(fontsize=24)
plt.savefig('../images/ROC_no_thyroid.png')


############## REMOVE PHYSICAL ACTIVITY ##############
# MAKE A PLOT OF BMI VS EXERCISE

X_train_3 = X_train.drop(['eth_african_american', 'eth_asian',
       'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'physical_activity_some', 'physical_activity_moderate','physical_activity_vigorous'], axis = 1)
X_test_3 = X_test.drop(['eth_african_american', 'eth_asian',
       'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'physical_activity_some', 'physical_activity_moderate','physical_activity_vigorous'], axis = 1)



model_sm_3 = sm.Logit(y_train, sm.add_constant(X_train_3))

print(model_sm_3.fit().summary())


model_3 = LogisticRegression(class_weight='balanced')
model_3.fit(X_train_3.values, y_train.values)
probabilities = model_3.predict(X_test_3)
probs = model_3.predict_proba(X_test_3)[:, 1]

accuracy_3 = metrics.accuracy_score(y_test, probabilities)
precision_3 = metrics.precision_score(y_test, probabilities)
recall_3 = metrics.recall_score(y_test, probabilities)
f1_score_3 = metrics.f1_score(y_test, probabilities)


fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
auc_3 = metrics.roc_auc_score(y_test, probs)

df_metrics.loc[len(df_metrics)] = [auc_3, accuracy_3, precision_3, recall_3,f1_score_3, 'no physical act']


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
ax.plot(fpr, tpr, color='b', lw=2, label='Model')
ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Postive Rate", fontsize=20)
ax.set_title("ROC curve: Ethnicity/Thyroid/PA Removed", fontsize=24)
ax.text(0.3, 0.7, " ".join(["AUC:",str(auc_3.round(3))]), fontsize=20)
ax.legend(fontsize=24)
plt.savefig('../images/ROC_no_physact.png')

######### keeping only significant predictors ######

X_train_4 = X_train[['age', 'bmi', 'piv_yes']]
X_test_4 = X_test[['age', 'bmi', 'piv_yes']]

model_sm_4 = sm.Logit(y_train, sm.add_constant(X_train_4))

print(model_sm_4.fit().summary())


model_4 = LogisticRegression(class_weight='balanced')
model_4.fit(X_train_4.values, y_train.values)
probabilities = model_4.predict(X_test_4)
probs = model_4.predict_proba(X_test_4)[:, 1]

accuracy_4 = metrics.accuracy_score(y_test, probabilities)
precision_4 = metrics.precision_score(y_test, probabilities)
recall_4 = metrics.recall_score(y_test, probabilities)
f1_score_4 = metrics.f1_score(y_test, probabilities)


fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
auc_4 = metrics.roc_auc_score(y_test, probs)

df_metrics.loc[len(df_metrics)] = [auc_4, accuracy_4, precision_4, recall_4,f1_score_4, 'only sig']

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
ax.plot(fpr, tpr, color='b', lw=2, label='Model')
ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Postive Rate", fontsize=20)
ax.set_title("ROC curve: only age/bmi/piv ", fontsize=24)
ax.text(0.3, 0.7, " ".join(["AUC:",str(auc_4.round(3))]), fontsize=20)
ax.legend(fontsize=24)
plt.savefig('../images/ROC_age_bmi_piv.png')

######### keeping only significant predictors ######

X_train_5 = X_train[['age', 'piv_yes']]
X_test_5 = X_test[['age', 'piv_yes']]

model_sm_5 = sm.Logit(y_train, sm.add_constant(X_train_5))

print(model_sm_5.fit().summary())


model_5 = LogisticRegression(class_weight='balanced')
model_5.fit(X_train_5.values, y_train.values)
probabilities = model_5.predict(X_test_5)
probs = model_5.predict_proba(X_test_5)[:, 1]

accuracy_5 = metrics.accuracy_score(y_test, probabilities)
precision_5 = metrics.precision_score(y_test, probabilities)
recall_5 = metrics.recall_score(y_test, probabilities)
f1_score_5 = metrics.f1_score(y_test, probabilities)

fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
auc_5 = metrics.roc_auc_score(y_test, probs)

df_metrics.loc[len(df_metrics)] = [auc_5, accuracy_5, precision_5, recall_5,f1_score_5, 'age/piv']

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
ax.plot(fpr, tpr, color='b', lw=2, label='Model')
ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Postive Rate", fontsize=20)
ax.set_title("ROC curve: only age/piv ", fontsize=24)
ax.text(0.3, 0.7, " ".join(["AUC:",str(auc_5.round(3))]), fontsize=20)
ax.legend(fontsize=24)
plt.savefig('../images/ROC_age_piv.png')
