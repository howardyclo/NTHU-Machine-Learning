import os
import sys
import datetime
import argparse
import numpy as np
import pandas as pd

from utils import *
from regression_nn import RegressionNN

parser = argparse.ArgumentParser(description='Nerual network (NN) for regression.')
parser.add_argument('--header_filename', dest='header_filename', type=str, default='energy_efficiency_cooling_load_training_header.csv', help='Training dataset with header csv. This is used to output hypothesis. (Default: "energy_efficiency_cooling_load_training_header.csv")')
parser.add_argument('--train_filename', dest='train_filename', type=str, default='energy_efficiency_cooling_load_training.csv', help='Training dataset csv. (Default: "energy_efficiency_cooling_load_training.csv")')
parser.add_argument('--test_filename', dest='test_filename', type=str, default='energy_efficiency_cooling_load_testing.csv', help='Training dataset csv. (Default: "energy_efficiency_cooling_load_testing.csv")')
parser.add_argument('--depth', dest='depth', type=int, default=2, help='Number of weight matrices (Defualt: 2. (depth-1) hidden layers)')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.01, help='Step size of updating weights. (Default: 0.01)')
parser.add_argument('--reg_lambda', dest='reg_lambda', type=float, default=0.0, help='Strength of L2 regularization for weights. (Default: 0.0)')
parser.add_argument('--K', dest='K', type=int, default=None, help='Denotes for "K"-fold cross-validation for determine the optimal hyper-parameters for NN. (Default: None)')
parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=30000, help='Max iteration for NN training algorithm to avoid not converging. (Defualt: 30000)')

args = parser.parse_args()

if __name__ == '__main__':
    # Check file paths exist
    header_filepath = 'dataset/' + args.header_filename
    train_filepath = 'dataset/' + args.train_filename
    test_filepath = 'dataset/' + args.test_filename

    if not os.path.exists(header_filepath):
        print('[!] File "{}" does not exist.'.format(header_filepath))
        print('[!] Exit program.')
        sys.exit(0)
    if not os.path.exists(train_filepath):
        print('[!] File "{}" does not exist.'.format(train_filepath))
        print('[!] Exit program.')
        sys.exit(0)
    if not os.path.exists(test_filepath):
        print('[!] File "{}" does not exist.'.format(test_filepath))
        print('[!] Exit program.')
        sys.exit(0)

    # Load dataset.
    try:
        data = load_dataset(train_filepath=train_filepath, test_filepath=test_filepath, classification=False)
    except Exception as e:
        print('[!] {}.'.format(e))
        print('[!] Exit program.')
        sys.exit(0)

    # Perform grid search K-fold cross validation.
    if args.K:

        log_filename = 'logs/SGD-{}-fold-[{}].txt'.format(args.K, datetime.datetime.now().strftime('%H:%M:%S'))

        # Specify list of hyper-parameters for grid search K-fold cross validation.
        param_grid = {
            'depth': [2, 3, 4, 5, 6],
            'reg_lambda': [0.0, 0.001, 0.01],
            'max_iteration': [args.max_iteration]
        }

        cv = GridSearchCV(data=data, model=RegressionNN, param_grid=param_grid, num_folds=args.K)
        cv.train()

        # Print and write CV results to log file.
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Cross validation history:')
        for cv_result in cv.cv_results:
            print_and_write(filename=log_filename, log=' -  Parameter: {} | Cross validation error: {}'.format(cv_result['param'], cv_result['score']))
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Best parameter: {}'.format(cv.best_param))
        print_and_write(filename=log_filename, log='[*] Best cross validation error: {}'.format(cv.best_score))
        print_and_write(filename=log_filename, log='[*] Start to train on full training data and evaluate on test data ...')

        # Train on full training data and evaluate on test data with the best hyper-parameters.
        model = RegressionNN(data=data, param=cv.best_param)
        model.train()

        # Compute test error.
        test_error = model.evaluation(X=data['test_X'], y=data['test_y'])

        # Print and write results to log file.
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Datetime: {}'.format(timestamp))
        print_and_write(filename=log_filename, log='[*] Train file path: "{}"'.format(train_filepath))
        print_and_write(filename=log_filename, log='[*] Test file path: "{}"'.format(test_filepath))
        print_and_write(filename=log_filename, log='[*] Best parameter: {}'.format(cv.best_param))
        print_and_write(filename=log_filename, log='[*] Network shape: {}'.format(model.network.shapes))
        print_and_write(filename=log_filename, log='[*] Performance: Test error: {}'.format(test_error))
        print_and_write(filename=log_filename, log='-'*100)

        # Output SGD hypothesis (need training data with header)
        hypothesis_filepath = 'hypothesis/SGD_hypothesis_header-[{}].csv'.format(timestamp)
        print_and_write(filename=log_filename, log='[*] Saving SGD hypothesis to "{}" ...'.format(hypothesis_filepath))
        try:
            model.to_csv(filepath=hypothesis_filepath)
            print_and_write(filename=log_filename, log='[*] Output SGD hypothesis to "{}" success.'.format(hypothesis_filepath))
        except Exception as e:
            print_and_write(filename=log_filename, log='[!] {}.'.format(e))
            print_and_write(filename=log_filename, log='[!] Output SGD hypothesis failed.')
        print_and_write(filename=log_filename, log='-'*100)

    else:
        # Specify hyper-parameters
        param = {
            'depth': args.depth,
            'reg_lambda': args.reg_lambda,
            'max_iteration': args.max_iteration
        }

        # Train on full training data and evaluate on test data with the specified hyper-parameters
        model = RegressionNN(data=data, param=param)

        model.train(info='Train on specified parameter: {}'.format(param))

        # Compute training and test error.
        test_error = model.evaluation(X=data['test_X'], y=data['test_y'])

        # Print results
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print('-'*100)
        print('[*] Datetime: {}'.format(timestamp))
        print('[*] Train file path: "{}"'.format(train_filepath))
        print('[*] Test file path: "{}"'.format(test_filepath))
        print('[*] Specified parameter: {}'.format(param))
        print('[*] Network shape: {}'.format(model.network.shapes))
        print('[*] Performance: Test error: {}'.format(test_error))
        print('-'*100)

        # Output SGD hypothesis (need training data with header)
        hypothesis_filepath = 'hypothesis/SGD_hypothesis_header-[{}].csv'.format(timestamp)
        print('[*] Saving SGD hypothesis to "{}" ...'.format(hypothesis_filepath))
        try:
            model.to_csv(filepath=hypothesis_filepath)
            print('[*] Output SGD hypothesis to "{}" success.'.format(hypothesis_filepath))
        except Exception as e:
            print('[!] {}'.format(e))
            print('[!] Output SGD hypothesis failed.')
        print('-'*100)
