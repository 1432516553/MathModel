import os
import matplotlib
import numpy as no
import pandas as pd


# CV ranking
cv_feature_importance = [(column, np.std(data[column])/abs(np.mean(data[column]))) for column in features]
cv_feature_importance.sort(key=lambda x: x[1], reverse=True)
cv_sorted_attrs = [item[0] for item in cv_feature_importance]
cv_ranking = [cv_sorted_attrs.index(x) + 1 for x in features]
print("# CV result")
print(cv_ranking)
print(cv_feature_importance)


# Pearson ranking
from scipy.stats import pearsonr

y_values = data[target]
pearsonr_feature_importance = [(column, pearsonr(data[column], y_values)[0]) for column in features]
pearsonr_feature_importance.sort(key=lambda x: abs(x[1]), reverse = True)
pearsonr_sorted_attrs = [item[0] for item in pearsonr_feature_importance]
pearsonr_ranking = [pearsonr_sorted_attrs.index(x) + 1 for x in features]
print("# Pearson result")
print(pearsonr_ranking)
print(pearsonr_feature_importance)


# RF(Random Forest) feature ranking
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=500, n_jobs=-1)
rf.fit(data.ix[:, features], y_values)
rf_scores = rf.feature_importances_
print(rf_scores)
rf_result = sorted(list(zip(features, rf_scores)), key=operator, itemgetter(1), reverse=True)
rf_sorted_attrs = [x[0] for x in rf_result]
rf_ranking = [rf_sorted_attrs.index(x) + 1 for x in features]
print("# Random forest result")
print(rf_sorted_attrs)
print(rf_ranking)


# MIC(Maximal Information Coefficient)feature ranking
from minepy import MINE

mine = MINE()
mic_scores = []
data_s = data.sample(frac=0.001)
print(data_s, shape)
for attr in features:
    print(attr)
mine.compute_score(data_s.loc[:, attr], y_values)
mic_scores.append(mine.mic())
mic_result = sorted(list(zip(features, mic_scores)), key=operator.itemgetter, reverse=True)
mic_sorted_attrs = [x[0] for x in mic_result]
mic_ranking = [mic_sorted_attrs.index(x) + 1 for x in featuers]
print("# MIC result")
print(mic_sorted_attrs)
print(mic_ranking)



# Catboost feature ranking
import catboost as cat
crf = cat.CatBoostRegressor(
    iterations = 10000,
    depth = 10,
    learning_rate = 0.02,
    verbose = 100,
    loss_function = 'RMES',
    eval_metric = 'RMES',
    early_stopping_rounds = 100,
    random_seed = 2019,
    task_type = 'GPU',
    bootstrap_type = 'Poisson',
    devices = '3'
)
crf.fit(train_data, eval_set = eval_data)
cat_ranking = [(y, x) for x, y in zip(data, columns, crf.feature_importance_)]
cat_ranking.sort()
print(cat_ranking)