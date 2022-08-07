from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from train_test_data import data_import, x_y_train_test_psm

import statistics
import pandas as pd
import geopandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import date

def outputs(joined_data_csv, test_size, max_depth, output_directory = 'outputs', site_name = 'restored_2006', random_state=42, cv=3):
    data_with_pft = data_import(joined_data_csv)
    data = data_with_pft.replace(to_replace=['short_grass', 'dead_grass_mix', 'water', 'long_grass', 'shrub_sphagnum', 'pool_bogbean', 'brash', 'rushes', 'bare', 'grass_sphagnum', 'sitka_pine', 'agri_grasses', 'calluna'], value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    logistic_regression_score = logistic_regression(data)
    rf_test_score = random_forest(data_with_pft, test_size, max_depth, output_directory, site_name = site_name, random_state=random_state, cv=cv)
    # decision_tree = decision_tree()
    # svm = svm()

    return logistic_regression_score, rf_test_score


# def svm(data, test_size, max_depth, output_directory, site_name = 'restored_2006', random_state=42, cv=3):
#     # write code - using the random_forest as a basis
#
#
# def decision_tree(data, test_size, max_depth, output_directory, site_name = 'restored_2006', random_state=42, cv=3):
#     # write code - using the random_forest as a basis



def random_forest(data, test_size, max_depth, output_directory, site_name = 'restored_2006', random_state=42, cv=3):
    predictor_train, predicted_train, predictor_test, predicted_test = x_y_train_test_psm(data, test_size)
    predictor_train = predictor_train.values.reshape(-1, 1)
    predictor_test = predictor_test.values.reshape(-1, 1)
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(predictor_train, predicted_train)
    clf_score = cross_val_score(clf, predictor_test, predicted_test, cv=cv)
    clf_pred_full = clf.predict(data[data.columns[3]]).values.reshape(-1, 1)
    formatted_date = date.today().strftime('%Y-%m-%d-%H%M%S')
    clf_pred_full.to_csv(
        f"{output_directory}/{site_name}-{formatted_date}.csv")
    clf_pred_full.to_file(
        f"{output_directory}/{site_name}-{formatted_date}.shp")
    return clf_score

# issue is that x is a string
def logistic_regression(data):
    # data = data_import(data).replace(
    #     to_replace=['short_grass', 'dead_grass_mix', 'water', 'long_grass', 'shrub_sphagnum', 'pool_bogbean', 'brash',
    #                 'rushes', 'bare', 'grass_sphagnum', 'sitka_pine', 'agri_grasses', 'calluna'],
    #     value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    predictor = data[data.columns[3]]
    predicted = data[data.columns[2]]
    predictor = predictor.values.reshape(-1,1)
    clf = LogisticRegression(random_state=0).fit(predictor, predicted)
    clf_score = clf.score(predictor, predicted)

    return clf_score


# issues - can't produce multiple histograms
def data_descriptions(joined_data_csv):
    data = data_import(joined_data_csv)
    pfts = data['PFT'].unique()
    descriptive_stats = {}
    histograms = []

    for pft in pfts:
        df = data.loc[data['PFT'] == pft, 'Velocity']
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
