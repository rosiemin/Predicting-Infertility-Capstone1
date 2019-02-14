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



df, test_df = md.open_data()
X, y, df_nodum, df_full = clean.clean_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y)



#### TESTING TEST DATA FROM TRAIN TEST SPLIT 2
X_train_3 = X_train.drop(['eth_african_american', 'eth_asian',
      'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'physical_activity_some', 'physical_activity_moderate','physical_activity_vigorous','piv_yes'], axis = 1)

X_test_1 = X_test.drop(['eth_african_american', 'eth_asian',
      'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'physical_activity_some', 'physical_activity_moderate','physical_activity_vigorous','piv_yes'], axis = 1)

model_sm = sm.Logit(y_test, sm.add_constant(X_test_1))

print(model_sm.fit().summary())
#std train data here
model_final = LogisticRegression(class_weight={0:1, 1:10})
model_final.fit(X_train_3.values, y_train.values)

# use same scaler to transform test X

probs = model_final.predict_proba(X_test_1)[:, 1]
probabilities = np.where(probs >= 0.5, 1, 0)
probs1 = model_final.predict_proba(X_train_3)[:, 1]
probabilities1 = np.where(probs1 >= 0.5, 1, 0)

accuracy_final = metrics.accuracy_score(y_test, probabilities)
precision_final = metrics.precision_score(y_test, probabilities)
recall_final = metrics.recall_score(y_test, probabilities)
f1_score_final = metrics.f1_score(y_test, probabilities)

fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
fpr1, tpr1, thresholds1 = metrics.roc_curve(y_train, probs1)

auc_final = metrics.roc_auc_score(y_test, probs)
auc_train = metrics.roc_auc_score(y_train, probs1)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
ax.plot(fpr, tpr, color='b', lw=2, label='Test Model')
ax.plot(fpr1, tpr1, color='r', lw=2, label='Train Model')

ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Postive Rate", fontsize=20)
ax.set_title("ROC curve: Final Model", fontsize=24)
ax.text(0.2, 0.7, " ".join(["AUC:",str(auc_final.round(3))]), fontsize=20, color = 'b')
ax.text(0.2, 0.6, " ".join(['AUC:',str(auc_train.round(3))]), fontsize=20, color = 'r')
ax.legend(fontsize=24)

plt.savefig('../images/ROC_final.png')

conf_y_true = y_test.values
conf_y_true_lab = np.where(conf_y_true == 0, 'fertile', 'infertile')
conf_y_pro = probabilities
conf_y_pro_lab = np.where(conf_y_pro == 0, 'fertile', 'infertile')


def generate_confusion_matrix(y_test, y_pred, labels, title, filename, show=False):
    cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10,6))
    strings = np.asarray([['TP', 'FN'],
                                    ['FP', 'TN']])

    label = (np.asarray(["{1}\n({0})".format(string, value)
                      for string, value in zip(strings.flatten(),
                                               cm.flatten())])).reshape(2, 2)
    ax = sns.heatmap(df_cm, annot=label,fmt="", cmap='Blues_r', annot_kws={'size':16},cbar = False)
    plt.ylabel("Actual Label", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ttl = ax.title
    ttl.set_position([0.5, 1.03])
    plt.savefig(filename)

    if show:
        plt.show()

generate_confusion_matrix(conf_y_true_lab, conf_y_pro_lab, labels = ['infertile', 'fertile'], title = "Confusion Matrix of Final Model", filename = '../images/confusion_mat.png', show = False)

############ running it on the final holdout:
X_holdout, y_holdout, df_nodum_test, df_full_test = clean.clean_data(test_df)

X_holdout = X_holdout.drop(['eth_african_american', 'eth_asian',
      'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'physical_activity_some', 'physical_activity_moderate','physical_activity_vigorous', 'piv_yes'], axis = 1)

probs_h = model_final.predict_proba(X_holdout)[:, 1]
probabilities_h = np.where(probs_h >= 0.5, 1, 0)
probs1_h = model_final.predict_proba(X_train_3)[:, 1]
probabilities1_h = np.where(probs1_h >= 0.5, 1, 0)

accuracy_final_holdout = metrics.accuracy_score(y_holdout, probabilities_h)
precision_final_holdout = metrics.precision_score(y_holdout, probabilities_h)
recall_final_holdout = metrics.recall_score(y_holdout, probabilities_h)
f1_score_final_holdout = metrics.f1_score(y_holdout, probabilities_h)

fpr_h, tpr_h, thresholds_h = metrics.roc_curve(y_holdout, probs_h)
fpr1_h, tpr1_h, thresholds1_h = metrics.roc_curve(y_train, probs1_h)

auc_final_holdout = metrics.roc_auc_score(y_holdout, probs_h)
auc_train_holdout = metrics.roc_auc_score(y_train, probs1_h)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
ax.plot(fpr_h, tpr_h, color='b', lw=2, label='Holdout Model')
ax.plot(fpr1_h, tpr1_h, color='r', lw=2, label='Train Model')

ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Postive Rate", fontsize=20)
ax.set_title("ROC curve: Holdout Model", fontsize=24)
ax.text(0.2, 0.7, " ".join(["AUC:",str(auc_final_holdout.round(3))]), fontsize=20, color = 'b')
ax.text(0.2, 0.6, " ".join(['AUC:',str(auc_train_holdout.round(3))]), fontsize=20, color = 'r')
ax.legend(fontsize=24)

plt.savefig('../images/ROC_holdout.png')


conf_y_true = y_holdout.values
conf_y_true_lab = np.where(conf_y_true == 0, 'fertile', 'infertile')
conf_y_pro = probabilities_h
conf_y_pro_lab = np.where(conf_y_pro == 0, 'fertile', 'infertile')
generate_confusion_matrix(conf_y_true_lab, conf_y_pro_lab, labels = ['infertile', 'fertile'], title = "Confusion Matrix of Final Holdout Data", filename = '../images/confusion_mat_holdout.png', show = True)

print(recall_final)
print(recall_final_holdout)

########## proportional bar chart

df_nodum_cat = df_nodum[['sti','irr_periods', 'smoke']]

# df_longdum = pd.wide_to_long(df_nodum_cat)
# var = df_nodum_cat.groupby(['sti','piv', 'irr_periods', 'smoke'])
# df['proportion'] = df['value']/df.groupby(['Color','variable'])['value'].transform('sum')
x_var = ['STIs',  'Irregular Periods', 'Smoking Status']
sti_arr = df_nodum_cat['sti'].value_counts()/len(df_nodum_cat)
# piv_arr = df_nodum_cat['piv'].value_counts()/len(df_nodum_cat)
periods_arr = df_nodum_cat['irr_periods'].value_counts()/len(df_nodum_cat)
smoke_arr = df_nodum_cat['smoke'].value_counts()/len(df_nodum_cat)
#
# first_ = np.array([])
#
# ind = [x for x, _ in enumerate(x_var)]
