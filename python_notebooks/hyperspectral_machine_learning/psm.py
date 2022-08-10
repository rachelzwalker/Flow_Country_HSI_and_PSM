from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import scipy.stats
import numpy as np
import collections

from train_test_data import data_import, x_y_train_test_psm, x_y_train_test_psm_all_fields

import statistics
import pandas as pd
import geopandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import date
import random


# def sub_section_outputs

def outputs(joined_data_csv, field_name, test_size, max_depth, kernel='rbf', num_clusters=10, output_directory='outputs',
            site_name='restored_2006',
            random_state=42, cv=3):
    data_with_pft = data_import(joined_data_csv)
    data = data_with_pft.replace(
        to_replace=['short_grass', 'dead_grass_mix', 'water', 'long_grass', 'shrub_sphagnum', 'pool_bogbean',
                    'rushes', 'grass_sphagnum'],
        value=[1, 2, 3, 4, 5, 6, 7, 8])
    # to_replace=['short_grass', 'dead_grass_mix', 'water', 'long_grass', 'shrub_sphagnum', 'pool_bogbean', 'brash',
    #             'rushes', 'bare', 'grass_sphagnum', 'sitka_pine', 'agri_grasses', 'calluna'],
    # value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    predictor_train, predicted_train, predictor_test, predicted_test = train_test_on_pft(data_with_pft,
                                                                                         test_size=test_size)

    # predictor_train, predicted_train, predictor_test, predicted_test = x_y_train_test_psm_all_fields(data_with_pft, test_size=test_size)

    logistic_regression_score = logistic_regression(data)
    rf_test_score = random_forest(data_with_pft, field_name, predictor_train, predicted_train, predictor_test, predicted_test,
                                  max_depth, output_directory, site_name=site_name,
                                  random_state=random_state, cv=cv)
    decision_tree_score = decision_tree(data_with_pft, field_name, predictor_train, predicted_train, predictor_test, predicted_test,
                                        max_depth, output_directory, site_name=site_name,
                                        cv=cv)
    svm_score = svm(data_with_pft, field_name, predictor_train, predicted_train, predictor_test, predicted_test, kernel=kernel,
                    output_directory=output_directory,
                    site_name=site_name, cv=cv)

    full_data_kmc = data[data.columns[3]].values.reshape(-1, 1)
    kmc_description = kmc(full_data_kmc, field_name, data_with_pft, num_clusters, output_directory, site_name)
    # make results into dictionary
    return ('logistic regression: ' + str(logistic_regression_score)), (
            'random_forest_score, mean and sd: ' + str(rf_test_score)), (
                   'decision_tree_score: ' + str(decision_tree_score)), ('svm: ' + str(svm_score)), (
               'kmc description: '), kmc_description


def kmc(full_data_kmc, field_name, data_to_add_to, num_clusters, output_directory, site_name):
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(full_data_kmc)
    clusters_df = pd.DataFrame(clusters, columns=['Cluster'])
    data_to_add_to['Cluster'] = clusters_df
    model = 'k-means cluster'
    formatted_date = date.today().strftime('%Y-%m-%d-%H%M%S')
    data_to_add_to.to_csv(
        f"{output_directory}/{field_name}-{site_name}-{model}-{formatted_date}.csv")
    data_to_add_to.to_file(
        f"{output_directory}/{field_name}-{site_name}-{model}-{formatted_date}.shp")
    return data_to_add_to.describe()


def train_test_on_pft(data, test_size):
    predictor_train, predicted_train, predictor_test, predicted_test = x_y_train_test_psm(data, test_size)
    predictor_train = predictor_train.values.reshape(-1, 1)
    predictor_test = predictor_test.values.reshape(-1, 1)
    return predictor_train, predicted_train, predictor_test, predicted_test


def svm(data, field_name, predictor_train, predicted_train, predictor_test, predicted_test, kernel, output_directory,
        site_name='restored_2006', cv=3):
    svc = SVC(kernel=kernel)
    svc.fit(predictor_train, predicted_train)
    svc_score = cross_val_score(svc, predictor_test, predicted_test, cv=cv)
    full_data_to_map = data[data.columns[3]].values.reshape(-1, 1)
    # full_data_to_map = data[data.columns[3:7]]
    svc_pred_full = svc.predict(full_data_to_map)
    model_results = data[data.columns[0:4]]
    model_results = model_results.assign(svc_pred_full=svc_pred_full)
    model = 'SVM'
    formatted_date = date.today().strftime('%Y-%m-%d-%H%M%S')
    model_results.to_csv(
        f"{output_directory}/{field_name}-{site_name}-{model}-{formatted_date}.csv")
    model_results.to_file(
        f"{output_directory}/{field_name}-{site_name}-{model}-{formatted_date}.shp")
    return svc_score


def decision_tree(data, field_name, predictor_train, predicted_train, predictor_test, predicted_test, max_depth, output_directory,
                  site_name='restored_2006', cv=3):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(predictor_train, predicted_train)
    clf_score = cross_val_score(clf, predictor_test, predicted_test, cv=cv)
    full_data_to_map = data[data.columns[3]].values.reshape(-1, 1)
    # full_data_to_map = data[data.columns[3:7]]
    clf_pred_full = clf.predict(full_data_to_map)
    model_results = data[data.columns[0:4]]
    model_results = model_results.assign(clf_pred_full=clf_pred_full)
    formatted_date = date.today().strftime('%Y-%m-%d-%H%M%S')
    model = 'decision_tree'
    model_results.to_csv(
        f"{output_directory}/{field_name}-{max_depth}-{site_name}-{model}-{formatted_date}.csv")
    model_results.to_file(
        f"{output_directory}/{field_name}-{max_depth}-{site_name}-{model}-{formatted_date}.shp")
    return clf_score


def random_forest(data, field_name, predictor_train, predicted_train, predictor_test, predicted_test, max_depth, output_directory,
                  site_name='restored_2006', random_state=42, cv=3):
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(predictor_train, predicted_train)
    clf_score = cross_val_score(clf, predictor_test, predicted_test, cv=cv)
    full_data_to_map = data[data.columns[3]].values.reshape(-1, 1)
    # full_data_to_map = data[data.columns[3:7]]
    clf_pred_full = clf.predict(full_data_to_map)
    model_results = data[data.columns[0:4]]
    model_results = model_results.assign(clf_pred_full=clf_pred_full)

    formatted_date = date.today().strftime('%Y-%m-%d-%H%M%S')
    model = 'random_forest'
    model_results.to_csv(
        f"{output_directory}/{field_name}-{max_depth}-{site_name}-{model}-{formatted_date}.csv")
    model_results.to_file(
        f"{output_directory}/{field_name}-{max_depth}-{site_name}-{model}-{formatted_date}.shp")
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


def similarity_measures_velocity_pfts(joined_data_csv, field_name):
    dfs_dict = df_for_similarity_measures_field_pfts(joined_data_csv, field_name)
    regressions = []
    # list_1 = comparision_df[0]
    # list_2 = comparision_df[1]
    # regression = scipy.stats.pearsonr(list_1, list_2)
    # regressions.append(regression)

    # for pft in pfts:
    #     list_1 = comparision_df[pft]
    #     list_2 = comparision_df[pft+1]
    #     regression = scipy.stats.pearsonr(list_1, list_2)
    #     regressions.append(regression)

    for df in dfs_dict.values():
        corr = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, cmap="Blues", annot=True, ax=ax)
        # regressions.append(heatmap)

    # return regressions


def df_for_similarity_measures_field_pfts(joined_data_csv, field_name):
    histogram_data, descriptive_df = data_descriptions(joined_data_csv, field_name)
    pft_to_number_of_field = pft_to_number_field(joined_data_csv, field_name)

    df_dict = {}

    pfts_to_ignore = []

    sorted_dict = collections.OrderedDict(sorted(pft_to_number_of_field.items(), key=lambda x: x[1]))
    for pft_name, length in sorted_dict.items():
        comparison_df = pd.DataFrame()
        for pft, field in histogram_data.items():
            if pft in pfts_to_ignore:
                continue

            comparison_df[pft] = random.sample(field, k=length)

        df_dict[pft_name] = comparison_df
        pfts_to_ignore.append(pft_name)

    return df_dict


# def different_sized_dataframes(joined_data_csv):
#     sorted_list_lengths = sorted_list(joined_data_csv)
#     histogram_data, descriptive_df = data_descriptions(joined_data_csv)
#     pfts = histogram_data.keys()
#     df_dict = {}
#
#     for


def pft_to_number_field(joined_data_csv, field_name):
    histogram_data, descriptive_df = data_descriptions(joined_data_csv, field_name)
    pft_to_number_field = {}
    for pft, values in histogram_data.items():
        field = []
        for value in values:
            field.append(value)
        len_list = len(field)
        pft_to_number_field[pft] = len_list
    # sorted_dictionary = collections.OrderedDict(sorted(pft_to_number_field.items()))

    # pft_list_lengths_zipping = zip(pft, len_list)
    # pft_list_lengths = list(pft_list_lengths_zipping)

    # return type(sorted_dictionary) # type comes out as a collection rather than a dictionary
    return pft_to_number_field


def data_descriptions(joined_data_csv, field_name='Velocity'):
    data = data_import(joined_data_csv)
    pfts = data['PFT'].unique()
    descriptive_stats = {}
    histograms = []
    histogram_data = {}

    for pft in pfts:
        data_field = data.loc[data['PFT'] == pft, field_name].to_numpy('float').tolist()
        mean = statistics.mean(data_field)
        variance = statistics.variance(data_field)
        stdev = statistics.stdev(data_field)

        descriptive_stats[pft] = {'mean': mean, 'variance': variance, 'standard_deviation': stdev}
        fig, ax = plt.subplots()
        histogram = sns.histplot(data=data_field, ax=ax).set(title=pft)
        histograms.append(histogram)

        histogram_data.update({pft: data_field})

    descriptive_df = pd.DataFrame.from_dict(descriptive_stats)

    return histogram_data, descriptive_df


def overall_data_description(joined_data_csv):
    data = data_import(joined_data_csv)
    description = data.describe()
    return description
