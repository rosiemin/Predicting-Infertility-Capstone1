import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from matplotlib.colors import LinearSegmentedColormap


def pair_sns(df):
    df_nodum_cont['Fertility Status'] = np.where(df_nodum_cont['fert_stat']=='no', 'Fertile', 'Infertile')
    df_nodum_cont.drop('fert_stat', inplace = True, axis = 1)

    pplot = sns.pairplot(df_nodum_cont, hue = 'Fertility Status', palette={'Fertile':"navy", 'Infertile':"salmon"}, plot_kws={"s": 27, 'alpha': 0.5}, diag_kind='kde')

    pplot.savefig('../images/pairplot.png')


def box_cat_plot(df):
    df_nodum_cont = df[['fert_stat','age', 'bmi', 'alcohol']]
    df_nodum_cont['Fertility Status'] = np.where(df_nodum_cont['fert_stat']=='no', 'Fertile', 'Infertile')
    df_nodum_cont.drop('fert_stat', inplace = True, axis = 1)
    fig, axs = plt.subplots(ncols=3)
    sns.boxplot(x = "Fertility Status", y = "age", data = df_nodum_cont, palette={'Fertile':"navy", 'Infertile':"salmon"}, ax = axs[0]).xaxis.label.set_visible(False)
    sns.boxplot(x = "Fertility Status", y = "bmi", data = df_nodum_cont,palette={'Fertile':"navy", 'Infertile':"salmon"}, ax = axs[1]).xaxis.label.set_visible(False)
    sns.boxplot(x = "Fertility Status", y = "alcohol", data = df_nodum_cont,palette={'Fertile':"navy", 'Infertile':"salmon"}, ax = axs[2]).xaxis.label.set_visible(False)
    plt.xlabel("Fertility Status")
    plt.tight_layout()

    plt.savefig('../images/boxplot_by.png')

def prop_bar(df):
    df_nodum_cat = df[['sti','irr_periods', 'smoke']]
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
    plt.bar(ind, proportion_former, width=0.6, label='Former', color='gray', bottom=proportion_no + proportion_yes)
    plt.bar(ind, proportion_yes, width=0.6, label='Yes', color='salmon', bottom=proportion_no)
    plt.bar(ind, proportion_no, width=0.6, label='No', color='navy')

    plt.xticks(ind, x_var, fontsize = 14)
    plt.ylabel("Percent (%)", fontsize = 16)
    plt.xlabel("Categorical Parameters", fontsize = 16)
    plt.title("Proportional Percent of Categorical Parameters in Training Data", fontsize = 14)
    plt.ylim=1.0
    plt.legend(loc='upper center', bbox_to_anchor=(1.03, -0.020))

    plt.savefig('../images/proportionalcats.png')


def ROC_c(fpr, tpr, fpr1, tpr1, auc, auc1):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Luck')
    ax.plot(fpr, tpr, color='salmon', lw=2, label='Holdout Model')
    ax.plot(fpr1, tpr1, color='navy', lw=2, label='Train Model')

    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Postive Rate", fontsize=20)
    ax.set_title("ROC curve: Holdout Model", fontsize=24)
    ax.text(0.15, 0.65, " ".join(["AUC:",str(auc.round(3))]), fontsize=20, color = 'salmon')
    ax.text(0.15, 0.6, " ".join(['AUC:',str(auc1.round(3))]), fontsize=20, color = 'navy')
    ax.legend(fontsize=24)

    plt.savefig('../images/ROC_holdout.png')

colors = ['#08006d','#f6695f','#f6695f','#f6695f','#f6695f','#f6695f','#f6695f','#f6695f','#08006d']

boundaries = [0.0, 0.3,0.7, 1.0]  # custom boundaries

# here I generated twice as many colors,
# so that I could prune the boundaries more clearly
# hex_colors = sns.light_palette('navy', n_colors=len(boundaries) * 2 + 2, as_cmap=False).as_hex()
# hex_colors = [hex_colors[i] for i in range(0, len(hex_colors), 2)]
custom_color_map = LinearSegmentedColormap.from_list(
    name='custom_plt',
    colors=colors)
# p = sns.palplot(sns.color_palette(colors, n_colors=5))


def generate_confusion_matrix(y_test, y_pred, labels, title, filename, show=False):
    cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10,10))
    strings = np.asarray([['TP', 'FN'],
                                    ['FP', 'TN']])

    label = (np.asarray(["{1}\n({0})".format(string, value)
                      for string, value in zip(strings.flatten(),
                                               cm.flatten())])).reshape(2, 2)
    ax = sns.heatmap(df_cm, annot=label,fmt="", cmap=custom_color_map, annot_kws={'size':20},cbar = False)
    plt.ylabel("Actual Label", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ttl = ax.title
    ttl.set_position([0.5, 1.03])
    plt.savefig(filename)

    if show:
        plt.show()
