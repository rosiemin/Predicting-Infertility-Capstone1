import clean_data1516 as clean
import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import merge_data1516 as md
import ggplot
import statsmodels.formula.api as sm


df, test_df = md.open_data()
X, y = clean.clean_data(df)

######## EDA FIRST ##########

# examine relationships between my continuous variables and my outcomes excluding id

ax, fig
