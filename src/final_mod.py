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
plt.style.use('ggplot')



df, test_df = md.open_data()
X, y, df_nodum, df_full = clean.clean_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y)



#### TESTING TEST DATA FROM TRAIN TEST SPLIT 2
X_train_3 = X_train.drop(['eth_african_american', 'eth_asian',
      'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'physical_activity_some', 'physical_activity_moderate','physical_activity_vigorous','piv_yes'], axis = 1)

X_test_1 = X_test.drop(['eth_african_american', 'eth_asian',
      'eth_mexican_hispanic', 'eth_mixed_race', 'eth_other_hispanic', 'thyroid_yes', 'physical_activity_some', 'physical_activity_moderate','physical_activity_vigorous','piv_yes'], axis = 1)


#std train data here
model_final = LogisticRegression(class_weight='balanced')
model_final.fit(X_train_3.values, y_train.values)

# use same scaler to transform test X

probs = model_final.predict_proba(X_test_1)[:, 1]
probabilities = np.where(probs >= 0.4, 1, 0)
probs1 = model_final.predict_proba(X_train_3)[:, 1]
probabilities1 = np.where(probs1 >= 0.4, 1, 0)

accuracy_final = metrics.accuracy_score(y_test, probabilities)
precision_final = metrics.precision_score(y_test, probabilities)
recall_final = metrics.recall_score(y_test, probabilities)
recall_tr = metrics.recall_score(y_train, probabilities1)
f1_score_final = metrics.f1_score(y_test, probabilities)

fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
fpr1, tpr1, thresholds1 = metrics.roc_curve(y_train, probs1)

auc_final = metrics.roc_auc_score(y_test, probs)
auc_train = metrics.roc_auc_score(y_train, probs1)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
ax.plot(fpr, tpr, color='salmon', lw=2, label='Test Model')
ax.plot(fpr1, tpr1, color='navy', lw=2, label='Train Model')

ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Postive Rate", fontsize=20)
ax.set_title("ROC curve: Final Model", fontsize=24)
ax.text(0.15, 0.65, " ".join(["AUC:",str(auc_final.round(3))]), fontsize=20, color = 'salmon')
ax.text(0.15, 0.6, " ".join(['AUC:',str(auc_train.round(3))]), fontsize=20, color = 'navy')
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
    ax = sns.heatmap(df_cm, annot=label,fmt="", cmap='bwr', annot_kws={'size':16},cbar = False)
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

model_sm = sm.Logit(y_holdout, sm.add_constant(X_holdout))

print(model_sm.fit().summary())

probs_h = model_final.predict_proba(X_holdout)[:, 1]
probabilities_h = np.where(probs_h >= 0.4, 1, 0)
probs1_h = model_final.predict_proba(X_train_3)[:, 1]
probabilities1_h = np.where(probs1_h >= 0.4, 1, 0)

accuracy_final_holdout = metrics.accuracy_score(y_holdout, probabilities_h)
precision_final_holdout = metrics.precision_score(y_holdout, probabilities_h)
recall_final_holdout = metrics.recall_score(y_holdout, probabilities_h)
f1_score_final_holdout = metrics.f1_score(y_holdout, probabilities_h)
recall_train = metrics.recall_score(y_train, probabilities1_h)

fpr_h, tpr_h, thresholds_h = metrics.roc_curve(y_holdout, probs_h)
fpr1_h, tpr1_h, thresholds1_h = metrics.roc_curve(y_train, probs1_h)

auc_final_holdout = metrics.roc_auc_score(y_holdout, probs_h)
auc_train_holdout = metrics.roc_auc_score(y_train, probs1_h)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
ax.plot(fpr_h, tpr_h, color='salmon', lw=2, label='Holdout Model')
ax.plot(fpr1_h, tpr1_h, color='navy', lw=2, label='Train Model')

ax.set_xlabel("False Positive Rate", fontsize=20)
ax.set_ylabel("True Postive Rate", fontsize=20)
ax.set_title("ROC curve: Holdout Model", fontsize=24)
ax.text(0.15, 0.65, " ".join(["AUC:",str(auc_final_holdout.round(3))]), fontsize=20, color = 'salmon')
ax.text(0.15, 0.6, " ".join(['AUC:',str(auc_train_holdout.round(3))]), fontsize=20, color = 'navy')
ax.legend(fontsize=24)

plt.savefig('../images/ROC_holdout.png')


conf_y_true = y_holdout.values
conf_y_true_lab = np.where(conf_y_true == 0, 'fertile', 'infertile')
conf_y_pro = probabilities_h
conf_y_pro_lab = np.where(conf_y_pro == 0, 'fertile', 'infertile')
generate_confusion_matrix(conf_y_true_lab, conf_y_pro_lab, labels = ['infertile', 'fertile'], title = "Confusion Matrix of Final Holdout Data", filename = '../images/confusion_mat_holdout.png', show = False)

print(recall_tr)
print(recall_train)
print(recall_final)
print(recall_final_holdout)

########## proportional bar chart ############

df_nodum_cat = df_nodum[['sti','irr_periods', 'smoke']]
x_var = ['STIs',  'Irregular Periods', 'Smoking Status']
sti_arr = df_nodum_cat['sti'].value_counts()
periods_arr = df_nodum_cat['irr_periods'].value_counts()
smoke_arr = df_nodum_cat['smoke'].value_counts()
ind = [x for x, _ in enumerate(x_var)]
no_ = np.array([sti_arr[0], periods_arr[0],smoke_arr[0]])
yes_ = np.array([sti_arr[1], periods_arr[1],smoke_arr[1]])
former_ = np.array([0, 0, smoke_arr[2]])

total = len(df_nodum_cat)

proportion_no = np.true_divide(no_, total) * 100
proportion_yes = np.true_divide(yes_, total) * 100
proportion_former = np.true_divide(former_, total) * 100

plt.figure(figsize=(10,8))
plt.bar(ind, proportion_former, width=0.6, label='Former Smoker', color='gray', bottom=proportion_no + proportion_yes)
plt.bar(ind, proportion_yes, width=0.6, label='Yes or Current Smoker', color='salmon', bottom=proportion_no)
plt.bar(ind, proportion_no, width=0.6, label='No or Never Smoker', color='navy')

plt.xticks(ind, x_var, fontsize = 14)
plt.ylabel("Percent (%)", fontsize = 16)
plt.xlabel("Categorical Parameters", fontsize = 16)
plt.title("Proportional Percent of Categorical Parameters in Training Data", fontsize = 14)
plt.ylim=1.0
plt.legend(loc='upper center', bbox_to_anchor=(1, -0.034))

plt.savefig('../images/proportionalcats.png')

################## Continuous boxplot with distributions #############
df_nodum_cont = df_nodum[['fert_stat','age', 'bmi', 'alcohol']]
df_nodum_cont['Fertility Status'] = np.where(df_nodum_cont['fert_stat']=='no', 'Fertile', 'Infertile')
df_nodum_cont.drop('fert_stat', inplace = True, axis = 1)

pplot = sns.pairplot(df_nodum_cont, hue = 'Fertility Status', palette={'Fertile':"navy", 'Infertile':"salmon"}, plot_kws={"s": 27, 'alpha': 0.5})
pplot.savefig('../images/pairplot.png')

fig, axs = plt.subplots(ncols=3)

sns.boxplot(x = "Fertility Status", y = "age", data = df_nodum_cont, palette={'Fertile':"navy", 'Infertile':"salmon"},boxprops={'alpha':.5}, ax = axs[0]).xaxis.label.set_visible(False)
sns.boxplot(x = "Fertility Status", y = "bmi", data = df_nodum_cont,palette={'Fertile':"navy", 'Infertile':"salmon"},boxprops={'alpha':.5}, ax = axs[1]).xaxis.label.set_visible(False)
sns.boxplot(x = "Fertility Status", y = "alcohol", data = df_nodum_cont,palette={'Fertile':"navy", 'Infertile':"salmon"},boxprops={'alpha':.5}, ax = axs[2]).xaxis.label.set_visible(False)
plt.xlabel("Fertility Status")
plt.tight_layout()
plt.savefig('../images/boxplot_by.png')
