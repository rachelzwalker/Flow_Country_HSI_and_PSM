from confusion_matrix import merge_each_output_with_test_train, data_import, data_to_input, confusion_outputs_2006_only

datasets = [
    data_to_input('restored_2006', '../outputs/restored_2006_full_fw-second derivative-full-0.25-4-3-2022-08-09-000000.csv'),
    ]


train_test_dataset = [
    data_to_input('train_test', '../train_test_data.csv')
]

if __name__ == "__main__":
    confusion_outputs_2006_only(datasets, train_test_dataset, 'test')

