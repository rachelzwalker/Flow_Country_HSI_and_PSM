from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from train_test_data import data_import, x_y_train_test_psm

import statistics
import pandas as pd
import geopandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import date


def outputs(joined_data_csv, test_size, max_depth, output_directory='outputs', site_name='restored_2006',
            random_state=42, cv=3):
    data_with_pft = data_import(joined_data_csv)
    data = data_with_pft.replace(
        to_replace=['short_grass', 'dead_grass_mix', 'water', 'long_grass', 'shrub_sphagnum', 'pool_bogbean', 'brash',
                    'rushes', 'bare', 'grass_sphagnum', 'sitka_pine', 'agri_grasses', 'calluna'],
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    predictor_train, predicted_train, predictor_test, predicted_test = train_test_on_pft(data_with_pft,
                                                                                         test_size=test_size)
    logistic_regression_score = logistic_regression(data)
    rf_test_score = random_forest(data_with_pft, predictor_train, predicted_train, predictor_test, predicted_test,
                                  max_depth, output_directory, site_name=site_name,
                                  random_state=random_state, cv=cv)
    decision_tree_score = decision_tree(data_with_pft, predictor_train, predicted_train, predictor_test, predicted_test,
                                        max_depth, output_directory, site_name=site_name,
                                        cv=cv)
    # svm = svm()

    return ('logistic regression' + str(logistic_regression_score)), (
                'random_forest_score, mean and sd' + str(rf_test_score)), (
                       'decision_tree_score' + str(decision_tree_score))


def train_test_on_pft(data, test_size):
    predictor_train, predicted_train, predictor_test, predicted_test = x_y_train_test_psm(data, test_size)
    predictor_train = predictor_train.values.reshape(-1, 1)
    predictor_test = predictor_test.values.reshape(-1, 1)
    return predictor_train, predicted_train, predictor_test, predicted_test


# def svm(data, test_size, max_depth, output_directory, site_name = 'restored_2006', random_state=42, cv=3):
#     # write code - using the random_forest as a basis
#
#
def decision_tree(data, predictor_train, predicted_train, predictor_test, predicted_test, max_depth, output_directory,
                  site_name='restored_2006', cv=3):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(predictor_train, predicted_train)
    clf_score = cross_val_score(clf, predictor_test, predicted_test, cv=cv)
    full_data_to_map = data[data.columns[3]].values.reshape(-1, 1)
    clf_pred_full = clf.predict(full_data_to_map)
    model_results = data[data.columns[0:4]]
    model_results = model_results.assign(clf_pred_full=clf_pred_full)
    formatted_date = date.today().strftime('%Y-%m-%d-%H%M%S')
    model = 'decision_tree'
    model_results.to_csv(
        f"{output_directory}/{site_name}-{model}-{formatted_date}.csv")
    model_results.to_file(
        f"{output_directory}/{site_name}-{model}-{formatted_date}.shp")
    return clf_score


def random_forest(data, predictor_train, predicted_train, predictor_test, predicted_test, max_depth, output_directory,
                  site_name='restored_2006', random_state=42, cv=3):
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(predictor_train, predicted_train)
    clf_score = cross_val_score(clf, predictor_test, predicted_test, cv=cv)
    full_data_to_map = data[data.columns[3]].values.reshape(-1, 1)
    clf_pred_full = clf.predict(full_data_to_map)
    model_results = data[data.columns[0:4]]
    model_results = model_results.assign(clf_pred_full=clf_pred_full)

    formatted_date = date.today().strftime('%Y-%m-%d-%H%M%S')
    model = 'random_forest'
    model_results.to_csv(
        f"{output_directory}/{site_name}-{model}-{formatted_date}.csv")
    model_results.to_file(
        f"{output_directory}/{site_name}-{model}-{formatted_date}.shp")
    return clf_score


# issue is that x is a string
def logistic_regression(data):
    # data = data_import(data).replace(
    #     to_replace=['short_grass', 'dead_grass_mix', 'water', 'long_grass', 'shrub_sphagnum', 'pool_bogbean', 'brash',
    #                 'rushes', 'bare', 'grass_sphagnum', 'sitka_pine', 'agri_grasses', 'calluna'],
    #     value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    predictor = data[data.columns[3]]
    predicted = data[data.columns[2]]
    predictor = predictor.values.reshape(-1, 1)
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
