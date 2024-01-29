#!/home/anirudh.nemmani/anaconda3/envs/misc/bin/python3
import pandas as pd
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

tot_fit = pd.read_csv('/home/anirudh.nemmani/Projects/misc/more_misc/final_fit.csv', index_col=0)
tot_fit

# input
X = tot_fit.iloc[:, [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]].values
# output
y = tot_fit.iloc[:, 33].values.astype('int')

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=0)

score_list = ['roc_auc', 'accuracy', 'average_precision', 'recall', 'f1', 'neg_log_loss']

steps_dt = [('over', SMOTE(sampling_strategy=1, random_state=0)), ('model', DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features=None, min_samples_leaf=5, min_samples_split=2 , random_state=0))]
pipeline_dt = Pipeline(steps=steps_dt)

param_dist = {
    'model__criterion': ['gini', 'entropy'],
    'model__splitter': ['best', 'random'],
    'model__max_depth': [1 ,2, 3, 4, 5, 6, 7, 8, 9],
    'model__min_samples_split': [2, 3, 4, 5, 6, 7, 8],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['auto', 'sqrt', 'log2', None]
}

grid_search = GridSearchCV(pipeline_dt, param_grid=param_dist, scoring=score_list, refit='roc_auc', cv=cv, n_jobs=30, return_train_score=True)
grid_search.fit(X, y)
best_params = grid_search.best_params_
print(best_params)

results_dict = {'Scoring Metric': [], 'Best Parameters': [], 'Best Score': [], 'Parameter Rankings': []}

# Loop over each scoring metric
for scorer in score_list:
    # Store the results for the current scoring metric
    results_dict['Scoring Metric'].append(scorer)
    results_dict['Best Parameters'].append(grid_search.best_params_)
    results_dict['Best Score'].append(grid_search.best_score_)

    # Get the parameter rankings for the scoring metric
    results = grid_search.cv_results_
    params_ranking = results[f'rank_test_{scorer}']
    param_names = results['params']
    param_rankings = [f"Rank {rank}" for rank in params_ranking]

    results_dict['Parameter Rankings'].append(list(zip(param_rankings, param_names)))

# Create a DataFrame from the results dictionary
results_df = pd.DataFrame(results_dict)

cv_results = pd.DataFrame.from_dict(grid_search.cv_results_)
list(cv_results.columns)
cv_results.to_csv('/home/anirudh.nemmani/Projects/misc/more_misc/HI_ML/results/grid_search_results_dt_redshift.csv')

df = cv_results[['params', 'rank_test_roc_auc', 'rank_test_accuracy', 'rank_test_average_precision']]

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('expand_frame_repr', False)
print(df)
df.to_csv('/home/anirudh.nemmani/Projects/misc/more_misc/HI_ML/results/grid_search_ranks_dt_redshift.csv')
