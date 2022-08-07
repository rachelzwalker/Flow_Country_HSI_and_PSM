from psm import logistic_regression, random_forest

if __name__ == "__main__":
    random_forest('../../InSAR/joined_file_test.csv', 0.3, 5)
