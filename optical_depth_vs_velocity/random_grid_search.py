import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from tqdm.notebook import tqdm

#intervening fit
i_c = pd.read_csv("intervening_fit.txt", sep='\t')
#intervening fit without nan or inf or -inf
i_c_2 = i_c.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

#getting all the file names which have nan or inf or -inf
all_i = i_c.merge(i_c_2.drop_duplicates(), on = ['Filename'], how='left', indicator=True)
intervening_non_fit = all_i[all_i['_merge'] == 'left_only']
# for i in intervening_non_fit['Filename'].tolist():
#     print(i)

#associated fit
a_c = pd.read_csv("associated_fit.txt", sep='\t')
#associated fit without nan or inf or -inf
a_c_2 = a_c.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

all_a = a_c.merge(a_c_2.drop_duplicates(), on = ['Filename'], how='left', indicator=True)
associated_non_fit_c = all_a[all_a['_merge'] == 'left_only']
# for i in associated_non_fit_c['Filename'].tolist():
#     print(i)

#adding coloumn int or ass to all fits
i_c_2['Class'] = '1'
a_c_2['Class'] = '0'

i_c_2.reset_index(drop=True, inplace=True)
a_c_2.reset_index(drop=True, inplace=True)

tot_fit = pd.concat([a_c_2, i_c_2])
tot_fit

# input
X = tot_fit.iloc[:, [7, 9, 11, 13, 15, 17, 21, 29, 31]].values #excluding 19, 23, 25, 27 columns
# output
y = tot_fit.iloc[:, 33].values.astype('int')

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=0)

score_list = ['roc_auc', 'accuracy', 'average_precision']

steps_rf = [('over', SMOTE(sampling_strategy=1, random_state=0)), ('model', RandomForestClassifier(n_estimators=7, max_depth=2, max_features='sqrt', min_samples_leaf=3, min_samples_split=2, random_state=0))]
pipeline_rf = Pipeline(steps=steps_rf)

param_dist = {
    'model__n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'model__max_depth': [3, 5, 7, 9],
    'model__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None],
    'model__bootstrap': [True, False]
}

# n_iter=100
# with tqdm(total=n_iter) as pbar:
#     random_search = RandomizedSearchCV(pipeline_rf, param_distributions=param_dist, n_iter=n_iter, scoring=score_list, refit='roc_auc', cv=cv, random_state=0)
#     for _ in range(n_iter):
#         random_search.fit(X, y)
#         pbar.update(1)

# Get the best hyperparameters from random search
random_search = RandomizedSearchCV(pipeline_rf, param_distributions=param_dist, n_iter=100, scoring=score_list, refit='roc_auc', cv=cv, random_state=0, n_jobs=-1, verbose=2)
random_search.fit(X, y)
best_params = random_search.best_params_
print(best_params)
