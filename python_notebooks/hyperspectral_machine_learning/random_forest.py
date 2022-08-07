from train_test_data import x_and_y_train_test, data_import

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from datetime import date

# from sklearn.metrics import confusion_matrix
END_COL_SPECTRUM_RANGE = [66, 76, 186, 359]
START_COL_SPECTRUM_RANGE = [6, 62, 66, 186]


def original_data_transformation(name, training_csv_path, complete_dataset_csv):
    return _data_transformation(name, training_csv_path, complete_dataset_csv, 362)


def derivative_data_transformation(name, training_csv_path, complete_dataset_csv):
    return _data_transformation(name, training_csv_path, complete_dataset_csv, 359)


def random_forest_csv_and_shape_file_outputs(site_name, data_transformations, test_size, max_depth, output_directory,
                                             random_state=42, cv=3):
    predictions = []

    for transformation in data_transformations:
        predictions.extend(
            random_forest_results(transformation, test_size, max_depth, random_state=random_state, cv=cv))

    predictions_view = pd.DataFrame(predictions,
                                    columns=['model_result', 'spectrum', 'data_transformation', 'accuracy_mean',
                                             'accuracy_standard_deviation'])
    top_results_to_convert = _random_forest_predictions_to_output(predictions_view)

    formatted_date = date.today().strftime('%Y-%m-%d-%H%M%S')
    top_results_to_convert.to_csv(f"{output_directory}/{site_name}-{test_size}-{max_depth}-{formatted_date}.csv")

    for index, row in top_results_to_convert.iterrows():
        row['model_result'].to_csv(f"{output_directory}/{site_name}-{row['data_transformation']}-{row['spectrum']}-{formatted_date}.csv")
        row['model_result'].to_file(f"{output_directory}/{site_name}-{row['data_transformation']}-{row['spectrum']}-{formatted_date}.shp")
    return predictions_view

# prior to the push, they were more variable


def random_forest_results(transformation, test_size, max_depth, random_state=42, cv=3):
    data_csv = transformation['training_csv_path']
    data_transformation = transformation['name']
    data = data_import(data_csv)

    x_train, y_train, x_test, y_test = x_and_y_train_test(data, test_size, x_start=transformation['start_band'],
                                                          x_end=transformation['end_band'], y=transformation['pft'])
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(x_train, y_train)
    test_score = cross_val_score(clf, x_test, y_test, cv=cv)

    full_data = data_import(transformation['complete_dataset_csv'])
    bands_for_prediction = full_data[full_data.columns[1:359]]
    clf_pred_full = clf.predict(bands_for_prediction)
    model_results = full_data[full_data.columns[359:361]]
    model_results_original = model_results.assign(clf_pred_full=clf_pred_full)

    spectrum_range = ['VIS', 'RE', 'NIR', 'SWIR']
    dict_outcome = [_outcome(model_results_original, test_score, 'full', data_transformation)]

    for index, spectral_range in enumerate(spectrum_range):
        x_train_sub = data_subset_spectrum_range_predictor(index, x_train)
        x_test_sub = data_subset_spectrum_range_predictor(index, x_test)
        clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
        clf.fit(x_train_sub, y_train)
        test_score = cross_val_score(clf, x_test_sub, y_test, cv=cv)

        x_sub = bands_for_prediction[bands_for_prediction.columns[START_COL_SPECTRUM_RANGE[index]:END_COL_SPECTRUM_RANGE[index]]]
        clf_pred_full = clf.predict(x_sub)
        model_results = full_data[full_data.columns[359:361]]
        model_results_all = model_results.assign(clf_pred_full=clf_pred_full)

        dict_outcome.append(_outcome(model_results_all, test_score, spectral_range, data_transformation))

    return dict_outcome


def data_subset_spectrum_range_predictor(index, predictor):
    x_train_sub = predictor.iloc[:, START_COL_SPECTRUM_RANGE[index]:END_COL_SPECTRUM_RANGE[index]]
    return x_train_sub


def _data_transformation(name, training_csv_path, complete_dataset_csv, pft):
    return {
        'name': name,
        'training_csv_path': training_csv_path,
        'complete_dataset_csv': complete_dataset_csv,
        'start_band': 1,
        'end_band': 359,
        'pft': pft
    }


def _random_forest_predictions_to_output(predictions_view):
    return predictions_view.nlargest(5, 'accuracy_mean')


def _outcome(model_result, test_score, spectrum, transformation):
    return {
        'model_result': model_result,
        'spectrum': spectrum,
        'data_transformation': transformation,
        'accuracy_mean': test_score.mean(),
        'accuracy_standard_deviation': test_score.std()
    }
