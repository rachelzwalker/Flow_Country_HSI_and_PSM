To get started you must install the requirements via
```
pip install -r requirements.txt
```
Use at least Python 3

# Using the Library
py files are PyCharm files that can be run via Jupyter notebooks
Each file has a different function demonstrated by its name and summarised below
See inside files for the specific data input and output requirements
Some of the files depend on others to function

## Confusion_matrix.py
This py file is used in the 'Outputs' notebook
It compares two datasets with the same geometry and same PFT classes to output a confusion matrix containing totals and producer's and user's accuracies.
There are two parts of the code more likely to be used in notebooks:
### Example 1
Using the full train-test data (13 PFTs)
```
confusion_outputs(output_datasets, test_train_dataset, name, output_directory)
```

### Example 2
Using the focused train-test data (10 PFTs)
```
confusion_outputs_2006_only(output_datasets, test_train_dataset, name, output_directory) - this 
```

## psm.py
This py file is used in the 'Outputs' notebook
It performs machine learning classifications (logistic regression, decision tree, random forest and SVM) and k-means clustering on the joined data (combined the PSM attribute(s) with the PFT predictions in QGIS using r.neighbor to find closest value to the PFT data and join by location)
The inputted data is imported using `data_import` from `train_test_data.py`
The inputted data is split into training and testing data using `x_y_train_test_psm` or `x_y_train_test_psm_all_fields` from `train_test_data.py`

It also performs similarity measures, producing histograms and correlation martices: 
```
similarity_measures_velocity_pfts(joined_data_csv, field_name)
```
or histograpms and descriptive statistics: 
```
data_descriptions(joined_data_csv, field_name)
```


## random_forest.py
This py file is used in the `Random_Forest` notebook
It performs the random forest machine learning classification on hyperspectral data, using the reflectance of each wavelength to predict PFTs using trarining data.
The inputted data is imported using `data_import` from `train_test_data.py`
The inputted data is split into training and testing data using `x_and_y_train_test` from `train_test_data.py`
Multiple data transformations can be added to iterate through
Each transformation is used to classify predictions for five spectral ranges (the whole data set, visible, red-edge, NIR and SWIR parts of the spectrum)
The bands for each part of the spectrum are defined in `END_COL_SPECTRUM_RANGE` and `START_COL_SPECTRUM_RANGE`; these may need updating depending on the number of bands and wavelengths of the hyperspectral data
Outputs include shape files and csvs of five predictions with the highest mean accuracies and a csv with inforamtion regarding these most accurate outcomes (settings for the random forest, which transformation, which spectral range, mean accuracy and standard deviation)
The table with all results is returned, but not saved as an output


## train_test_data.py
This py is not used in isolation, but to aid machine learning
It can be used to import data into a geodataframe using the data_import function
It can be used to create the training and testing dataset with the required ratios for machine learning, however, the PFTs will be subject to change and need to be adapted.
The most likely funciton to be used to train the data is the 
```
x_and_y_train_test(data, test_size, x_start=1, x_end=359, y=359)
```
x (columns used to predict) and y (prediction) values are predetermined, but can be adapted as required


## Vector.py
This py is used to convert a raster dataset with many attributes to a vector dataset (outputted as a shapefile to view in GIS and as a csv for further analysis)
The function to be used is 
```
output_csv_for_analysis(data_tiff, geom_file, name) 
```