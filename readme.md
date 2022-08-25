# Flow Country HSI PSM
A hyperspectral approach to understand the association between PSM
(as measured by InSAR data) and land cover and substrate detail for a
Scottish peatland.

## Project Purpose
Add aim and objectives
Aim: To determine whether hyperspectral data can be used to understand the association between peatland surface motion (as measured by InSAR data) and land cover in the Flow Country.

Objectives:
1. Assess the extent to which supervised and unsupervised machine learning algorithms can be used to classify plant functional types.
2. Determine whether machine learning can be used to show a relationship between plant functional types and peat surface motion.

## Using the Notebooks
ipynb files are Jupyter notebook files and can be run as such
Each notebook has a different function demonstrated by its name and summarised below
See inside notebook files for the specific data input and output requirements
Some of the notebooks depend on the libraries inside the hyperspectral_machine_learning library
Use at least Python 3

### Continuum_removal.ipynb
This notebook iterates through the pre-processed hyperspectral data to create a new csv file containing the continuum removal data transformation.


### Derivatives.ipynb
This notebook iterates through the pre-processed hyperspectral data to create two new csv files containing the first and second derivative data transformations.


### k-means_clustering.ipynb
This notebook takes hyperspectral and Peat Surface Motion (PSM) data to be clustered with options to change the bands used to focus on specific parts of the spectrum - you would need to check if these bands match your data (visible: 400-700 nm, red-edge: 680-750 nm, NIR: 700-1300 nm and SWIR: 1300-2500 nm).
The `k` value can be changed easily to determine the number of clusters required.
A cluster map and csv are outputted.

### McNemar_calculation.ipynb
This notebook takes two confusion matrices to test for similarity.
The comparison returns the true positive (TP), tru negative (TN), false positive (FP) and false negative (FN) scores along with a z score.
If z is greater than **3.481**, then it is statistically significant diffierent at 0.05 level (1 degree of freedom chi2 table). i.e. the classifiers have statsitically significane different outcomes.
The number of columns will need to be taken into account, with changes made to the code as described in the notebook.

### Outputs.ipynb
This notebook can be used to create confusion matrices, run machine learning on joined PSM and PFT files and apply similarity measures.

The confusion matrices are outputted to a csv and accuracy scores returned in the notebook.
The number of datasets inputted can vary should any need to be merged.
The confusion matrix outputted contains the producer's and user's accuracies and the column and row totals.

The machine learning is run through the outputs function and takes a variety of inputs which are listed in the markdown.
The machine learning classifications applied to the data are logistic regression, decision tree, random forest and SVM. A k-means cluster is also outputted.
The PFT name inluded in the library are specific to this study (on a Scottish peatland) so would need to be adapted for use.

The similarity measures returns histograms using the name of the attribute inputted into the function to determine whcih graphs should be constructed. A series of correlation matrices are also returned to demonstrate whether there are stron correlations between PFTs and therefore whether some need to be merged prior to further analysis. If the PFT class sizes vary, a series of correlation martices are created, removing the smallest class each time, with larger classes reduced to the smallest size in each iteration (random values extracted from the larger class).

### PCA.ipynb
This notebook conducts PCA on the inputted hyperspectral data and produces a scatter graph for the relationship between the PFT classes with the two highest PCA outcomes. Variance is also returned as is a variance bar chart.
Numbers and list lengths need to be adapted depending on the number of classes in the input data.
The PFT name inluded in the notebook are specific to this study (on a Scottish peatland) so would need to be adapted for use.

### Random_Forest.ipynb
This notebook iterates through the different data transformations and spectral ranges (embedded into PyCharm code) to demonstrate which transformation/range combination produces the more accurate predictions.
The PFT name inluded in the library are specific to this study (on a Scottish peatland) so would need to be adapted for use.
The inputs can be changed to vary the ratio of train:test data, maximum depth size and cross validation.
The outputs include a shape file and csv of the location of each pixel and the predicted PFT for the top five outputs. A csv file containing the mean accuracies and standard deviations for the top five is also outputted and the complete table of mean accuracies and standard deviations is returned in the notebook.

### vector_creation.ipynb
This notebook is used to convert a raster dataset with many attributes to a vector dataset (outputted as a shapefile to view in GIS and as a csv for further analysis).
