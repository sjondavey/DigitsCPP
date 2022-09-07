import pandas as pd
import numpy as np
from pathlib import Path
from random import randrange

def get_training_labels_from_input(Y_as_labels):
    ''' takes a value, i, in the range [0,9] and converts it into a vector of length 10 with the i-th element 
        set to 1 and all the others set to 0 
    '''
    labelled_data = np.zeros((10, len(Y_as_labels)))
    for i in range(len(Y_as_labels)):
        labelled_data[Y_as_labels[i],i] = 1

    return labelled_data


def read_kaggle_data(filename, training_set_percentage, random_state = 200):
    all_data = pd.read_csv(filename)
    if (training_set_percentage < 0 or training_set_percentage > 1):
        raise ValueError("training_set_percentage must in the range [0, 1]")
    train_data = all_data.sample(frac= training_set_percentage, random_state = random_state) #random state is a seed value
    test_data = all_data.drop(train_data.index)
    # Input data needs to be normalized. I have chosen to make these normalized between 0 and 1
    X_train = train_data[train_data.columns[1:]].to_numpy().T / 256
    Y_train = get_training_labels_from_input(train_data['label'].to_numpy())
    X_test = test_data[test_data.columns[1:]].to_numpy().T / 256
    Y_test = get_training_labels_from_input(test_data['label'].to_numpy())
    return X_train, Y_train, X_test, Y_test

def read_kaggle_data_all_into_training(filename):
    all_data = pd.read_csv(filename)
    X_train = all_data[all_data.columns[1:]].to_numpy().T / 256
    Y_train = get_training_labels_from_input(all_data['label'].to_numpy())
    return X_train, Y_train


def read_parameters_from_file(inputpath, fileprefix):
    parameters = {}
    counter = 0
    constants_file = inputpath + fileprefix + "_b" + str(counter) + ".csv"
    weights_file = inputpath + fileprefix + "_W" + str(counter) + ".csv"
    file_exists = Path(constants_file).is_file()
    while (file_exists):
        parameters['W' + str(counter)] = pd.read_csv(weights_file, header = None).values
        parameters['b' + str(counter)] = pd.read_csv(constants_file, header = None).values
        counter = counter + 1
        constants_file = inputpath + fileprefix + "_b" + str(counter) + ".csv"
        weights_file = inputpath + fileprefix + "_W" + str(counter) + ".csv"
        file_exists = Path(constants_file).is_file()
    return parameters

def write_parameters_to_file(parameters, outputpath, fileprefix):
    counter = 0
    for i in range (0, len(parameters) // 2): # integer division
        constants_file = outputpath + fileprefix + "_b" + str(i) + ".csv"
        weights_file = outputpath + fileprefix + "_W" + str(i) + ".csv"
        np.savetxt(weights_file, parameters['W' + str(i)], delimiter=",")
        np.savetxt(constants_file, parameters['b' + str(i)], delimiter=",")


# for comparing data files with no headers. returns true / false value and an integer for the
# number of variables that have been checked to be 'equal' before the file was declared different
def compare_data_in_files(file1, file2):
    number_of_matches = 0
    data1 = pd.read_csv(file1).to_numpy()
    data2 = pd.read_csv(file2).to_numpy()
    if (data1.shape != data2.shape):
        return False, number_of_matches
    number_of_rows, number_of_columns = data1.shape
    rows_to_check = 5
    cols_to_check = 5
    for r in range(0,rows_to_check):
        row = randrange(number_of_rows)
        for c in range(0,cols_to_check):
            col = randrange(number_of_columns)
            if abs(data1[row,col] - data2[row,col]) > 1e-4:
                return False, number_of_matches
            else:
                number_of_matches = number_of_matches + 1
    return True, number_of_matches


