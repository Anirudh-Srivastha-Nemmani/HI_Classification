import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#intervening fit
i_c = pd.read_csv("busyfit_intervening.txt", sep='\t')
#intervening fit without nan or inf or -inf
i_c_2 = i_c.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

#getting all the file names which have nan or inf or -inf
all_i = i_c.merge(i_c_2.drop_duplicates(), on = ['Filename'], how='left', indicator=True)
intervening_non_fit = all_i[all_i['_merge'] == 'left_only']
# for i in intervening_non_fit['Filename'].tolist():
#     print(i)

#associated fit
a_c = pd.read_csv("busyfit_associated_corrected.txt", sep='\t')
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

steps_lr = [('scaler', StandardScaler()), ('over', SMOTE(sampling_strategy=1, random_state=0)), ('model', LogisticRegressionCV(Cs = 10, max_iter=5000, random_state=0, solver='liblinear', n_jobs=7))]
pipeline_lr = Pipeline(steps=steps_lr)
for i in score_list:
    print('{} for LR SMOTE Model is {}'.format(i, round(np.mean(cross_val_score(pipeline_lr, X, y, scoring=i, cv=cv, n_jobs=-1)),3)))

param_grid = {
    'model__penalty': ['l1', 'l2', 'elasticnet'],
    'model__Cs': [1, 10, 100],
    'model__solver': ['liblinear', 'saga'],
    'model__max_iter': [100, 300, 500, 700, 900]
}

# random = RandomizedSearchCV(pipeline_lr, param_grid, n_iter=100, scoring=score_list, cv=cv, refit="roc_auc", n_jobs=7, verbose=2, random_state=0)

# random.fit(X, y)
# print("Best hyperparameters: ", random.best_params_)
# print("ROC-AUC: ", random.best_score_)
# print("Accuracy: ", random.cv_results_["mean_test_accuracy"][random.best_index_])
# print("Precision: ", random.cv_results_["mean_test_average_precision"][random.best_index_])

grid_search = GridSearchCV(
    pipeline_lr, param_grid=param_grid, scoring=score_list, cv=cv, refit="roc_auc", n_jobs=6, verbose=2
)

grid_search.fit(X, y)
print("Best hyperparameters: ", grid_search.best_params_)
print("ROC-AUC: ", grid_search.best_score_)
print("Accuracy: ", grid_search.cv_results_["mean_test_accuracy"][grid_search.best_index_])
print("Precision: ", grid_search.cv_results_["mean_test_average_precision"][grid_search.best_index_])
