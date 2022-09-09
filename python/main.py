import time
import neural_network
from neural_network import *

import read_input_data
from read_input_data import *

import configparser

def read_config_file(config_file_name):
    path = Path(config_file_name)
    if (not path.is_file()):
        raise FileExistsError("File: " + config_file_name + " not found")
    config_parameters = {}
    config = configparser.ConfigParser()
    config.read(config_file_name)

    config_parameters['architecture'] = (config['network']['architecture']).split(',')
    config_parameters['learning_rate'] = float(config['network']['learning_rate'])
    config_parameters['use_existing_parameters'] = config.getboolean('network','use_existing_parameters')
    config_parameters['existing_parameters_path'] = config['network']['existing_parameters_path']
    config_parameters['existing_parameters_file_prefix'] = config['network']['existing_parameters_file_prefix']
    config_parameters['train'] = config.getboolean('data','train')
    config_parameters['epochs'] = int(config['data']['epochs'])
    config_parameters['input_data_file'] = config['data']['input_data_file']
    config_parameters['training_data_split'] = config.getfloat('data','training_data_split')
    config_parameters['write_trained_parameters_to_file'] = config.getboolean('data', 'write_trained_parameters_to_file')
    config_parameters['path_to_write_trained_parameters'] = config['data']['path_to_write_trained_parameters']
    config_parameters['parameter_file_prefix'] = config['data']['parameter_file_prefix']
    config_parameters['output_training_cost'] = config.getboolean('data', 'output_training_cost')
    config_parameters['output_cost_every_n_epochs'] = int(config['data']['output_cost_every_n_epochs'])

    return config_parameters




def main(config_parameters):    
    # this is not from the .ini file on purpose. If I use all the data for training, I do not want to 
    # randomize the order because I want to compare 
    if abs(config_parameters['training_data_split'] -1) < 1e-6:
        # in this case I do not want to 'randomise' the order of the data. This helps when regression testing against the C++ implementation
        X_train, Y_train = read_kaggle_data_all_into_training(config_parameters['input_data_file'])
    else:
        random_state = 200
        X_train, Y_train, X_test, Y_test = read_kaggle_data(config_parameters['input_data_file'], config_parameters['training_data_split'], random_state)

    if (config_parameters['use_existing_parameters']):
        parameters = read_parameters_from_file(config_parameters['existing_parameters_path'], config_parameters['existing_parameters_file_prefix'])
    else:
        parameters = get_random_weights(config_parameters['architecture'])

    if (config_parameters['train']):
        start = time.time()
        trained_parameters = train(config_parameters['architecture'], 
                                config_parameters['learning_rate'], 
                                parameters, X_train, Y_train, 
                                config_parameters['epochs'],
                                config_parameters['output_training_cost'],
                                config_parameters['output_cost_every_n_epochs'])
        end = time.time()
        print("Training time: " + str(end - start) + " seconds")

    if (config_parameters['write_trained_parameters_to_file']):
        write_parameters_to_file(parameters, config_parameters['path_to_write_trained_parameters'], config_parameters['parameter_file_prefix'])


if __name__ == "__main__":

    config_file = "E:/Code/kaggle/digits/examples/test.ini"
    config_parameters = read_config_file(config_file)

    # config_parameters['train'] = config.getboolean('data', 'train')
    # config_parameters['input_data_file'] = config['data']['input_data_file']

    

    main(config_parameters)
