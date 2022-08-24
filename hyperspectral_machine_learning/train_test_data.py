# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import geopandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix

def x_y_train_test_psm_all_fields(data, test_size):
    train, test = split_plant_functional_types_full(data, test_size)
    x_train = train[train.columns[3:7]]
    y_train = train[train.columns[2]]

    x_test = test[test.columns[3:7]]
    y_test = test[test.columns[2]]

    return x_train, y_train, x_test, y_test

def x_y_train_test_psm(data, test_size):
    train, test = split_plant_functional_types_full(data, test_size)
    x_train = train[train.columns[3]]
    y_train = train[train.columns[2]]

    x_test = test[test.columns[3]]
    y_test = test[test.columns[2]]

    return x_train, y_train, x_test, y_test


def x_and_y_train_test(data, test_size, x_start=1, x_end=359, y=359):
    train, test = split_plant_functional_types_full(data, test_size)

    x_train = train[train.columns[x_start:x_end]]
    y_train = train[train.columns[y]]

    x_test = test[test.columns[x_start:x_end]]
    y_test = test[test.columns[y]]

    return x_train, y_train, x_test, y_test


def x_and_y_train_test_sections(data_csv, test_size, x_start, x_end, y):
    train, test = split_plant_functional_types_sections(data_csv, test_size)

    x_train = train[train.columns[x_start:x_end]]
    y_train = train[train.columns[y]]

    x_test = test[test.columns[x_start:x_end]]
    y_test = test[test.columns[y]]

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


def split_plant_functional_types_full(data_frame, test_size):
    train_shrub_sphagnum, test_shrub_sphagnum = train_test_split(
        data_frame[data_frame['PFT'].str.contains('shrub_sphagnum')], test_size=test_size)
    train_water, test_water = train_test_split(data_frame[data_frame['PFT'].str.contains('water')],
                                               test_size=test_size)
    train_grass_sphagnum, test_grass_sphagnum = train_test_split(
        data_frame[data_frame['PFT'].str.contains('grass_sphagnum')], test_size=test_size)
    train_pool_bogbean, test_pool_bogbean = train_test_split(
        data_frame[data_frame['PFT'].str.contains('pool_bogbean')], test_size=test_size)
    train_calluna, test_calluna = train_test_split(
        data_frame[data_frame['PFT'].str.contains('calluna')], test_size=test_size)
    train_rushes, test_rushes = train_test_split(
        data_frame[data_frame['PFT'].str.contains('rushes')], test_size=test_size)
    train_long_grass, test_long_grass = train_test_split(
        data_frame[data_frame['PFT'].str.contains('long_grass')], test_size=test_size)
    train_short_grass, test_short_grass = train_test_split(
        data_frame[data_frame['PFT'].str.contains('short_grass')], test_size=test_size)
    train_brash, test_brash = train_test_split(
        data_frame[data_frame['PFT'].str.contains('brash')], test_size=test_size)
    train_dead_grass_mix, test_dead_grass_mix = train_test_split(
        data_frame[data_frame['PFT'].str.contains('dead_grass_mix')], test_size=test_size)
    train_bare, test_bare = train_test_split(
        data_frame[data_frame['PFT'].str.contains('bare')], test_size=test_size)
    train_sitka_pine, test_sitka_pine = train_test_split(
        data_frame[data_frame['PFT'].str.contains('sitka_pine')], test_size=test_size)
    train_agri_grasses, test_agri_grasses = train_test_split(
        data_frame[data_frame['PFT'].str.contains('agri_grasses')], test_size=test_size)

    train_frames = [train_shrub_sphagnum, train_water, train_grass_sphagnum, train_pool_bogbean, train_calluna,
                    train_rushes, train_long_grass, train_short_grass, train_brash, train_dead_grass_mix, train_bare, train_sitka_pine, train_agri_grasses]
    test_frames = [test_shrub_sphagnum, test_water, test_grass_sphagnum, test_pool_bogbean, test_calluna,
                   test_rushes, test_long_grass, test_short_grass, test_brash, test_dead_grass_mix, test_bare, test_sitka_pine, test_agri_grasses]

    # train_frames = [train_bare, train_brash, train_shrub_sphagnum, train_water, train_grass_sphagnum, train_pool_bogbean, train_rushes,
    #                 train_long_grass, train_short_grass, train_dead_grass_mix]
    # test_frames = [test_bare, test_brash, test_shrub_sphagnum, test_water, test_grass_sphagnum, test_pool_bogbean,
    #                test_rushes, test_long_grass, test_short_grass, test_dead_grass_mix]

    train_plant_functional_types = pd.concat(train_frames)
    test_plant_functional_types = pd.concat(test_frames)
    return train_plant_functional_types, test_plant_functional_types


def data_import(data_csv):
    data = pd.read_csv(data_csv)
    data['geometry'] = geopandas.GeoSeries.from_wkt(data['geometry'])
    data_gdf = geopandas.GeoDataFrame(data, geometry='geometry')
    return data_gdf

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
