import pandas as pd
from sklearn.metrics import confusion_matrix
from train_test_data import data_import
from sklearn.metrics import accuracy_score


def confusion_outputs_2006_only(output_datasets, test_train_dataset, name, output_directory='outputs'):
    merged_dataset_for_confusion_matrix = merge_each_output_with_test_train(output_datasets, test_train_dataset)
    sample = merged_dataset_for_confusion_matrix['Predicted PFT']
    prediction = merged_dataset_for_confusion_matrix['Sample PFT']
    accuracy = confusion_matrix_accuracy(sample, prediction)
    df_confusion = confusion_matrix(sample, prediction)
    producers_matrix = producers_accuracy_matrix_2006_only(df_confusion)
    matrix = users_accuracy_matrix_2006_only(producers_matrix)
    matrix.to_csv(f"{output_directory}/{name}.csv")

    return accuracy


def confusion_outputs(output_datasets, test_train_dataset, name, output_directory='outputs'):
    merged_dataset_for_confusion_matrix = merge_each_output_with_test_train(output_datasets, test_train_dataset)
    sample = merged_dataset_for_confusion_matrix['Predicted PFT']
    prediction = merged_dataset_for_confusion_matrix['Sample PFT']
    accuracy = confusion_matrix_accuracy(sample, prediction)
    df_confusion = confusion_matrix(sample, prediction)
    producers_matrix = producers_accuracy_matrix(df_confusion)
    matrix = users_accuracy_matrix(producers_matrix)
    matrix.to_csv(f"{output_directory}/{name}.csv")

    return accuracy

def users_accuracy_matrix_2006_only(producers_matrix):
    column_label = 'Row_Total'

    users_accuracy_0 = producers_matrix.iloc[0, 0] / producers_matrix.loc['bare', column_label]
    users_accuracy_1 = producers_matrix.iloc[1, 1] / producers_matrix.loc['brash', column_label]
    users_accuracy_2 = producers_matrix.iloc[2, 2] / producers_matrix.loc['dead_grass_mix', column_label]
    users_accuracy_3 = producers_matrix.iloc[3, 3] / producers_matrix.loc['grass_sphagnum', column_label]
    users_accuracy_4 = producers_matrix.iloc[4, 4] / producers_matrix.loc['long_grass', column_label]
    users_accuracy_5 = producers_matrix.iloc[5, 5] / producers_matrix.loc['pool_bogbean', column_label]
    users_accuracy_6 = producers_matrix.iloc[6, 6] / producers_matrix.loc['rushes', column_label]
    users_accuracy_7 = producers_matrix.iloc[7, 7] / producers_matrix.loc['short_grass', column_label]
    users_accuracy_8 = producers_matrix.iloc[8, 8] / producers_matrix.loc['shrub_sphagnum', column_label]
    users_accuracy_9 = producers_matrix.iloc[9, 9] / producers_matrix.loc['water', column_label]


    users_accuracy = [users_accuracy_0, users_accuracy_1,
                      users_accuracy_2,
                      users_accuracy_3,
                      users_accuracy_4,
                      users_accuracy_5, users_accuracy_6, users_accuracy_7, users_accuracy_8, users_accuracy_9,
                      'n/a', 'n/a']
    producers_matrix.loc['users_accuracy'] = users_accuracy
    return producers_matrix

def users_accuracy_matrix(producers_matrix):
    column_label = 'Row_Total'

    users_accuracy_0 = producers_matrix.iloc[0, 0] / producers_matrix.loc['agri_grasses', column_label]
    users_accuracy_1 = producers_matrix.iloc[1, 1] / producers_matrix.loc['bare', column_label]
    users_accuracy_2 = producers_matrix.iloc[2, 2] / producers_matrix.loc['brash', column_label]
    users_accuracy_3 = producers_matrix.iloc[3, 3] / producers_matrix.loc['calluna', column_label]
    users_accuracy_4 = producers_matrix.iloc[4, 4] / producers_matrix.loc['dead_grass_mix', column_label]
    users_accuracy_5 = producers_matrix.iloc[5, 5] / producers_matrix.loc['grass_sphagnum', column_label]
    users_accuracy_6 = producers_matrix.iloc[6, 6] / producers_matrix.loc['long_grass', column_label]
    users_accuracy_7 = producers_matrix.iloc[7, 7] / producers_matrix.loc['pool_bogbean', column_label]
    users_accuracy_8 = producers_matrix.iloc[8, 8] / producers_matrix.loc['rushes', column_label]
    users_accuracy_9 = producers_matrix.iloc[9, 9] / producers_matrix.loc['short_grass', column_label]
    users_accuracy_10 = producers_matrix.iloc[10, 10] / producers_matrix.loc['shrub_sphagnum', column_label]
    users_accuracy_11 = producers_matrix.iloc[11, 11] / producers_matrix.loc['sitka_pine', column_label]
    users_accuracy_12 = producers_matrix.iloc[12, 12] / producers_matrix.loc['water', column_label]

    users_accuracy = [users_accuracy_0, users_accuracy_1,
                      users_accuracy_2,
                      users_accuracy_3,
                      users_accuracy_4,
                      users_accuracy_5, users_accuracy_6, users_accuracy_7, users_accuracy_8, users_accuracy_9,
                      users_accuracy_10, users_accuracy_11, users_accuracy_12, 'n/a', 'n/a']
    producers_matrix.loc['users_accuracy'] = users_accuracy
    return producers_matrix


def producers_accuracy_matrix_2006_only(df_confusion):
    df_confusion.loc['Column_Total'] = df_confusion.sum(numeric_only=True, axis=0)
    df_confusion.loc[:, 'Row_Total'] = df_confusion.sum(numeric_only=True, axis=1)
    row_label = 'Column_Total'

    # tried to do a for loop, but couldn't get it to work

    producers_accuracy_0 = df_confusion.iloc[0, 0] / df_confusion.loc[row_label, 'bare']
    producers_accuracy_1 = df_confusion.iloc[1, 1] / df_confusion.loc[row_label, 'brash']
    producers_accuracy_2 = df_confusion.iloc[2, 2] / df_confusion.loc[row_label, 'dead_grass_mix']
    producers_accuracy_3 = df_confusion.iloc[3, 3] / df_confusion.loc[row_label, 'grass_sphagnum']
    producers_accuracy_4 = df_confusion.iloc[4, 4] / df_confusion.loc[row_label, 'long_grass']
    producers_accuracy_5 = df_confusion.iloc[5, 5] / df_confusion.loc[row_label, 'pool_bogbean']
    producers_accuracy_6 = df_confusion.iloc[6, 6] / df_confusion.loc[row_label, 'rushes']
    producers_accuracy_7 = df_confusion.iloc[7, 7] / df_confusion.loc[row_label, 'short_grass']
    producers_accuracy_8 = df_confusion.iloc[8, 8] / df_confusion.loc[row_label, 'shrub_sphagnum']
    producers_accuracy_9 = df_confusion.iloc[9, 9] / df_confusion.loc[row_label, 'water']

    producers_accuracy = [producers_accuracy_0, producers_accuracy_1, producers_accuracy_2,
                          producers_accuracy_3, producers_accuracy_4, producers_accuracy_5, producers_accuracy_6,
                          producers_accuracy_7, producers_accuracy_8, producers_accuracy_9,
                          'n/a']

    df_confusion.loc[:, 'producers_accuracy'] = producers_accuracy

    return df_confusion


def producers_accuracy_matrix(df_confusion):
    df_confusion.loc['Column_Total'] = df_confusion.sum(numeric_only=True, axis=0)
    df_confusion.loc[:, 'Row_Total'] = df_confusion.sum(numeric_only=True, axis=1)
    row_label = 'Column_Total'

    # tried to do a for loop, but couldn't get it to work

    producers_accuracy_0 = df_confusion.iloc[0, 0] / df_confusion.loc[row_label, 'agri_grasses']
    producers_accuracy_1 = df_confusion.iloc[1, 1] / df_confusion.loc[row_label, 'bare']
    producers_accuracy_2 = df_confusion.iloc[2, 2] / df_confusion.loc[row_label, 'brash']
    producers_accuracy_3 = df_confusion.iloc[3, 3] / df_confusion.loc[row_label, 'calluna']
    producers_accuracy_4 = df_confusion.iloc[4, 4] / df_confusion.loc[row_label, 'dead_grass_mix']
    producers_accuracy_5 = df_confusion.iloc[5, 5] / df_confusion.loc[row_label, 'grass_sphagnum']
    producers_accuracy_6 = df_confusion.iloc[6, 6] / df_confusion.loc[row_label, 'long_grass']
    producers_accuracy_7 = df_confusion.iloc[7, 7] / df_confusion.loc[row_label, 'pool_bogbean']
    producers_accuracy_8 = df_confusion.iloc[8, 8] / df_confusion.loc[row_label, 'rushes']
    producers_accuracy_9 = df_confusion.iloc[9, 9] / df_confusion.loc[row_label, 'short_grass']
    producers_accuracy_10 = df_confusion.iloc[10, 10] / df_confusion.loc[row_label, 'shrub_sphagnum']
    producers_accuracy_11 = df_confusion.iloc[11, 11] / df_confusion.loc[row_label, 'sitka_pine']
    producers_accuracy_12 = df_confusion.iloc[10, 12] / df_confusion.loc[row_label, 'water']

    producers_accuracy = [producers_accuracy_0, producers_accuracy_1, producers_accuracy_2,
                          producers_accuracy_3, producers_accuracy_4, producers_accuracy_5, producers_accuracy_6,
                          producers_accuracy_7, producers_accuracy_8, producers_accuracy_9, producers_accuracy_10,
                          producers_accuracy_11, producers_accuracy_12, 'n/a']

    df_confusion.loc[:, 'producers_accuracy'] = producers_accuracy

    return df_confusion


def confusion_matrix(sample, prediction):
    df_confusion = pd.crosstab(sample, prediction)
    return df_confusion


def confusion_matrix_accuracy(sample, prediction):
    accuracy = accuracy_score(sample, prediction)
    return accuracy


def merge_each_output_with_test_train(output_datasets, test_train_dataset):
    imported_datasets_combined = combine_output_csvs_in_dataframe(output_datasets)
    imported_test_train_dataset = data_import(test_train_dataset)
    reduced_train_test = imported_test_train_dataset[['geometry', 'PFT']]

    merged_on_geometry = pd.merge(imported_datasets_combined, reduced_train_test, on="geometry", how="right")
    merged_on_geometry = merged_on_geometry.rename(columns={"clf_pred_full": "Predicted PFT", "PFT": "Sample PFT"})

    return merged_on_geometry


def combine_output_csvs_in_dataframe(output_datasets):
    imported_datasets_combined_df = pd.DataFrame()
    imported_datasets_dict = combine_output_csvs_in_dictionary(output_datasets)
    for name, dataset in imported_datasets_dict.items():
        reduced_dataframe_to_remove_point_id_which_will_be_repeated = dataset[['geometry', 'clf_pred_full']]
        frames = [imported_datasets_combined_df, reduced_dataframe_to_remove_point_id_which_will_be_repeated]
        imported_datasets_combined_df = pd.concat(frames)

    return imported_datasets_combined_df


def combine_output_csvs_in_dictionary(output_datasets):
    imported_datasets = {}
    for dataset in output_datasets:
        imported_dataset = import_csv(dataset)
        imported_datasets.update(imported_dataset)

    return imported_datasets


def import_csv(dataset):
    data_csv = dataset['csv_path']
    data_name = dataset['name']
    data = data_import(data_csv)
    imported_dataset = {}
    imported_dataset[data_name] = data

    return imported_dataset


def data_to_input(name, csv_path):
    return {
        'name': name,
        'csv_path': csv_path}
