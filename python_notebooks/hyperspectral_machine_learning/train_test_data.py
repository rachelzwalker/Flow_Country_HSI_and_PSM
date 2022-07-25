# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import geopandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification



def random_forest(data_csv, test_size, max_depth, random_state=42, cv=3):
    x_train, y_train, x_test, y_test = x_and_y_train_test(data_csv, test_size)
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(x_train, y_train)
    clf_train_pred = clf.predict(x_train)
    clf_pred = clf.predict(x_test)
    scores = cross_val_score(clf, x_train, y_train, cv=cv)
    test_score = cross_val_score(clf, x_test, y_test, cv=cv)
    return "%0.2f accuracy with a standard deviation of %0.2f" % (test_score.mean(), test_score.std())


def x_and_y_train_test(data_csv, test_size):
    train, test = split_plant_functional_types(data_csv, test_size)

    x_train = train[train.columns[1:359]]
    y_train = train[train.columns[359]]

    x_test = test[test.columns[1:359]]
    y_test = test[test.columns[359]]

    return x_train, y_train, x_test, y_test

# def dict_to_gdf(geo_data_frame, test_size):
#     train_plant_functional_types, test_plant_functional_types = split_plant_functional_types(geo_data_frame, test_size)
#     train_df = pd.DataFrame([train_plant_functional_types])
#     test_df = pd.DataFrame([test_plant_functional_types])
#     # train_df['geometry'] = geopandas.GeoSeries.from_wkt(train_df['geometry'])
#     # train_gdf = geopandas.GeoDataFrame(train_df, geometry='geometry')
#     # test_df['geometry'] = geopandas.GeoSeries.from_wkt(test_df['geometry'])
#     # test_gdf = geopandas.GeoDataFrame(test_df, geometry='geometry')
#     return train_df, test_df

def train_geo_frame(geo_data_frame, type, test_size):
    return train_test_split(geo_data_frame[geo_data_frame['PFT'].str.contains(type)], test_size=test_size)


def split_plant_functional_types(data_csv, test_size):
    data_frame = data_import(data_csv)
    train_shrub_sphagnum, test_shrub_sphagnum = train_test_split(
        data_frame[data_frame['PFT'].str.contains('shrub_sphagnum')], test_size=test_size)
    train_water, test_water = train_test_split(data_frame[data_frame['PFT'].str.contains('water')],
                                               test_size=test_size)
    train_sphagnum_r, test_sphagnum_r = train_test_split(
        data_frame[data_frame['PFT'].str.contains('spahgnum_r')], test_size=test_size)
    train_pool_bogbean, test_pool_bogbean = train_test_split(
        data_frame[data_frame['PFT'].str.contains('pool_bogbean')], test_size=test_size)
    train_calluna_mix, test_calluna_mix = train_test_split(
        data_frame[data_frame['PFT'].str.contains('calluna_mix')], test_size=test_size)
    train_rushes_sedges, test_rushes_sedges = train_test_split(
        data_frame[data_frame['PFT'].str.contains('rushes_sedges')], test_size=test_size)

    train_frames = [train_shrub_sphagnum, train_water, train_sphagnum_r, train_pool_bogbean, train_calluna_mix,
                    train_rushes_sedges]
    test_frames = [test_shrub_sphagnum, test_water, test_sphagnum_r, test_pool_bogbean, test_calluna_mix,
                   test_rushes_sedges]

    train_plant_functional_types = pd.concat(train_frames)
    test_plant_functional_types = pd.concat(test_frames)
    return train_plant_functional_types, test_plant_functional_types

# def split_plant_functional_types(csv, test_size):
#     data_frame = pd.read_csv(csv)
#     train_plant_functional_types = []
#     test_plant_functional_types = []
#     train_PFT = []
#     test_PFT = []
#
#     plant_functional_types = ['shrub_sphagnum', 'water', 'spahgnum_r', 'pool_bogbean', 'calluna_mix', 'rushes_sedges']
#     for plant_functional_type in plant_functional_types:
#         train_PFT[plant_functional_type], test_PFT[plant_functional_type] = train_test_split(data_frame[data_frame['PFT'].str.contains(plant_functional_type)], test_size=test_size)
#         train_plant_functional_types.append(train_PFT)
#         test_plant_functional_types.append(test_PFT)
#     return train_plant_functional_types, test_plant_functional_types

# def split_plant_functional_types(csv, test_size):
#     data_frame = pd.read_csv(csv)
#     train_plant_functional_types = {}
#     test_plant_functional_types = {}
#
#     plant_functional_types = ['shrub_sphagnum', 'water', 'spahgnum_r', 'pool_bogbean', 'calluna_mix', 'rushes_sedges']
#     for plant_functional_type in plant_functional_types:
#         train_plant_functional_types[plant_functional_type], test_plant_functional_types[plant_functional_type] = train_test_split(data_frame[data_frame['PFT'].str.contains(plant_functional_type)], test_size=test_size)
#
#     return train_plant_functional_types, test_plant_functional_types

def data_import(data_csv):
    data = pd.read_csv(data_csv)
    data['geometry'] = geopandas.GeoSeries.from_wkt(data['geometry'])
    data_gdf = geopandas.GeoDataFrame(data, geometry='geometry')
    return data_gdf


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
