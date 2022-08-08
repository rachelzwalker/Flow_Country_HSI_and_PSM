from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import scipy.stats
import numpy as np

from train_test_data import data_import, x_y_train_test_psm

import statistics
import pandas as pd
import geopandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import date

# def sub_section_outputs

def outputs(joined_data_csv, test_size, max_depth, kernel='rbf', num_clusters=10, output_directory='outputs',
            site_name='restored_2006',
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
    svm_score = svm(data_with_pft, predictor_train, predicted_train, predictor_test, predicted_test, kernel=kernel,
                    output_directory=output_directory,
                    site_name=site_name, cv=cv)

    full_data_kmc = data[data.columns[3]].values.reshape(-1, 1)
    kmc_description = kmc(full_data_kmc, data_with_pft, num_clusters, output_directory, site_name)

    return ('logistic regression: ' + str(logistic_regression_score)), (
            'random_forest_score, mean and sd: ' + str(rf_test_score)), (
                   'decision_tree_score: ' + str(decision_tree_score)), ('svm: ' + str(svm_score)), (
               'kmc description: '), kmc_description


def kmc(full_data_kmc, data_to_add_to, num_clusters, output_directory, site_name):
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(full_data_kmc)
    clusters_df = pd.DataFrame(clusters, columns=['Cluster'])
    data_to_add_to['Cluster'] = clusters_df
    model = 'k-means cluster'
    formatted_date = date.today().strftime('%Y-%m-%d-%H%M%S')
    data_to_add_to.to_csv(
        f"{output_directory}/{site_name}-{model}-{formatted_date}.csv")
    data_to_add_to.to_file(
        f"{output_directory}/{site_name}-{model}-{formatted_date}.shp")
    return data_to_add_to.describe()


def train_test_on_pft(data, test_size):
    predictor_train, predicted_train, predictor_test, predicted_test = x_y_train_test_psm(data, test_size)
    predictor_train = predictor_train.values.reshape(-1, 1)
    predictor_test = predictor_test.values.reshape(-1, 1)
    return predictor_train, predicted_train, predictor_test, predicted_test


def svm(data, predictor_train, predicted_train, predictor_test, predicted_test, kernel, output_directory,
        site_name='restored_2006', cv=3):
    svc = SVC(kernel=kernel)
    svc.fit(predictor_train, predicted_train)
    svc_score = cross_val_score(svc, predictor_test, predicted_test, cv=cv)
    full_data_to_map = data[data.columns[3]].values.reshape(-1, 1)
    svc_pred_full = svc.predict(full_data_to_map)
    model_results = data[data.columns[0:4]]
    model_results = model_results.assign(svc_pred_full=svc_pred_full)
    model = 'SVM'
    formatted_date = date.today().strftime('%Y-%m-%d-%H%M%S')
    model_results.to_csv(
        f"{output_directory}/{site_name}-{model}-{formatted_date}.csv")
    model_results.to_file(
        f"{output_directory}/{site_name}-{model}-{formatted_date}.shp")
    return svc_score


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


def similarity_measures_velocity_pfts(joined_data_csv):
    histogram_data, descriptive_df = data_descriptions(joined_data_csv)
    pfts = histogram_data.keys()
    df = pd.DataFrame()

    for pft, values in histogram_data.items():
        pft_df = pd.DataFrame()
        velocity = []
        for value in values:
            velocity.append(value)
        pft_df[pft] = velocity
        df = pd.concat([df, pft_df], ignore_index=True, axis=1)

    df.columns = pfts
    return df



def find_max_list(joined_data_csv):
    histogram_data, descriptive_df = data_descriptions(joined_data_csv)
    list_lengths = []
    for pft, values in histogram_data.items():
        velocity = []
        for value in values:
            velocity.append(value)
        len_list = len(velocity)
        list_lengths.append(len_list) #returning length of the string name rather than the number of floats

    return max(list_lengths)

# def similarity_measures_velocity_pfts(joined_data_csv):
#     histogram_data, descriptive_df = data_descriptions(joined_data_csv)
#     list_of_tuples = [(pft,velocity) for pft,velocity in dict.items(histogram_data)]
#     list_of_lists = [list(x) for x in list_of_tuples]
#     regressions = []
#
#     # return to and look at alternatives for extracting data from a dictionary/comparing values across a dictionary
#
#     for pft in range(10):
#         list_1_unsplit = list_of_lists[pft]
#
#         list_2 = list_of_lists[pft+1]
#         # regression = scipy.stats.pearsonr(list_1, list_2)
#         # regressions.append(regression)
#
#     return list_1[1]


def data_descriptions(joined_data_csv):
    data = data_import(joined_data_csv)
    pfts = data['PFT'].unique()
    descriptive_stats = {}
    histograms = []
    histogram_data = {}

    for pft in pfts:
        velocity = data.loc[data['PFT'] == pft, 'Velocity']
        velocity_list = velocity.to_numpy('float').tolist()
        mean = statistics.mean(velocity)
        variance = statistics.variance(velocity)
        stdev = statistics.stdev(velocity)

        descriptive_stats[pft] = {'mean': mean, 'variance': variance, 'standard_deviation': stdev}
        fig, ax = plt.subplots()
        histogram = sns.histplot(data=velocity, ax=ax).set(title=pft)
        histograms.append(histogram)

        histogram_data.update({pft: velocity_list})


    descriptive_df = pd.DataFrame.from_dict(descriptive_stats)

    return histogram_data, descriptive_df


def overall_data_description(joined_data_csv):
    data = data_import(joined_data_csv)
    description = data.describe()
    return description
