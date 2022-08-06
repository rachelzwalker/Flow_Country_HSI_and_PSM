from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from train_test_data import data_import, x_y_train_test_psm

import statistics
import pandas as pd
import geopandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def random_forest(joined_data_csv, test_size, max_depth, random_state=42, cv=3):
    x_train, y_train, x_test, y_test = x_y_train_test_psm(joined_data_csv, test_size)
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(x_train, y_train)
    test_score = cross_val_score(clf, x_test, y_test, cv=cv)
    return test_score

# issue is that x is a string
def logistic_regression(joined_data_csv):
    data = data_import(joined_data_csv).replace(to_replace=['short_grass', 'dead_grass_mix', 'water', 'long_grass', 'shrub_sphagnum', 'pool_bogbean', 'brash', 'rushes', 'bare', 'grass_sphagnum', 'sitka_pine', 'agri_grasses', 'calluna'], value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    predictor = data[data.columns[3]]
    predicted = data[data.columns[2]]
    predictor = predictor.values.reshape(-1,1)
    clf = LogisticRegression(random_state=0).fit(predictor, predicted)
    clf_score = clf.score(predictor, predicted)

    return clf_score


# issues - can't produce multiple histograms
def data_descriptions(joined_data_csv):
    data = data_import(joined_data_csv)
    pfts = data['clf_pred_f'].unique()
    descriptive_stats = {}
    histograms = []

    for pft in pfts:
        df = data.loc[data['clf_pred_f'] == pft, 'Velocity']
        mean = statistics.mean(df)
        variance = statistics.variance(df)
        stdev = statistics.stdev(df)

        stats = {pft: {'mean': mean, 'variance': variance, 'standard_deviation': stdev}}
        descriptive_stats.update(stats)
        fig, ax = plt.subplots()
        histogram = sns.histplot(data=df, ax=ax).set(title=pft)
        histograms.append(histogram)

    descriptive_df = pd.DataFrame.from_dict(descriptive_stats)
    graph = sns.barplot(data=descriptive_df)

    return histograms


def overall_data_description(joined_data_csv):
    data = data_import(joined_data_csv)
    description = data.describe()
    return description
