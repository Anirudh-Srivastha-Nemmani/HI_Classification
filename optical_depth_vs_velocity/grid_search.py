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
   'model__n_estimators': [6, 7, 8, 9],
   'model__max_depth': [3, 5, 7, 9],
   'model__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
   'model__min_samples_leaf': [1, 2, 4, 6, 8],
   'model__max_features': ['sqrt', 'log2', None],
   'model__bootstrap': [True]
}

# Get the best hyperparameters from random search
grid_search = GridSearchCV(pipeline_rf, param_grid=param_dist, scoring=score_list, refit='roc_auc', cv=cv, n_jobs=-1, return_train_score=True, verbose=2)
grid_search.fit(X, y)
best_params = grid_search.best_params_
print(best_params)
print('##########################################################')
print('Printing the ranking of the parameter grid for each scorer')
print('##########################################################')


# Print the results for each scoring metric
for score in score_list:
    print("Scoring metric:", score)
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

    # Get the parameter rankings for the scoring metric
    results = grid_search.cv_results_
    params_ranking = results[f'rank_test_{score}']
    param_names = results['params']

    print("Parameter Rankings:")
    for rank, param in zip(params_ranking, param_names):
        print(f"Rank {rank}: {param}")
    print("\n")
