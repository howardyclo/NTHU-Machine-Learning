import os
import sys
import datetime
import argparse
import numpy as np
import pandas as pd

from adaboost import AdaboostClassifer
from utils import *

parser = argparse.ArgumentParser(description='Binary Adaboost classifer.')
parser.add_argument('--train_filename', dest='train_filename', type=str, default='alphabet_DU_training.csv', help='Training dataset csv. (Default: "alphabet_DU_training.csv")')
parser.add_argument('--test_filename', dest='test_filename', type=str, default='alphabet_DU_testing.csv', help='Training dataset csv. (Default: "alphabet_DU_testing.csv")')
parser.add_argument('--K', dest='K', type=int, default=None, help='Denotes for "K"-fold cross-validation for determine the optimal hyper-parameters for adaboost classifer. (Default: None)')
parser.add_argument('--T', dest='T', type=int, default=5, help='The maximum number of classifers at which boosting is terminated.')

args = parser.parse_args()

if __name__ == '__main__':

    # Check file paths exist
    train_filepath = 'dataset/' + args.train_filename
    test_filepath = 'dataset/' + args.test_filename

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
        data = load_dataset(train_filepath=train_filepath, test_filepath=test_filepath)
    except Exception as e:
        print('[!] {}.'.format(e))
        print('[!] Exit program.')
        sys.exit(0)

    # Perform grid search K-fold cross validation.
    if args.K:
        log_filename = 'logs/adaboost-{}-fold-[{}].txt'.format(args.K, datetime.datetime.now().strftime('%H:%M:%S'))
        # Print and write label mapping
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Class label mapping:')
        print_and_write(filename=log_filename, log=' -  {} => +1'.format(data['+1']))
        print_and_write(filename=log_filename, log=' -  {} => -1'.format(data['-1']))

        # Specify list of hyper-parameters for grid search K-fold cross validation.
        param_grid = {'num_classifer': range(1, 50+1)}

        cv = GridSearchCV(data=data, model=AdaboostClassifer, param_grid=param_grid, num_folds=args.K)
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
        model = AdaboostClassifer(data=data, param=cv.best_param, verbose=True)
        model.train()

        # Compute training and test error.
        train_error = zero_one_loss(y_truth=data['train_y'], y_pred=model.hypothesis(X=data['train_X']))
        test_error = zero_one_loss(y_truth=data['test_y'], y_pred=model.hypothesis(X=data['test_X']))

        # Print and write results to log file.
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Datetime: {}'.format(timestamp))
        print_and_write(filename=log_filename, log='[*] Train file path: "{}"'.format(train_filepath))
        print_and_write(filename=log_filename, log='[*] Test file path: "{}"'.format(test_filepath))
        print_and_write(filename=log_filename, log='[*] Best parameter: {}'.format(cv.best_param))
        print_and_write(filename=log_filename, log='[*] Best hypothesis:')
        print_and_write(filename=log_filename, log='{}'.format(model))
        print_and_write(filename=log_filename, log='[*] Performance: Train error: {} | Test error: {}'.format(train_error, test_error))
        print_and_write(filename=log_filename, log='-'*100)

        # Output Adaboost hypothesis (need training data with header)
        hypothesis_filepath = 'hypothesis/Adaboost_hypothesis_header-[{}].csv'.format(timestamp)
        print_and_write(filename=log_filename, log='[*] Saving Adaboost hypothesis to "{}" ...'.format(hypothesis_filepath))
        try:
            model.to_csv(filepath=hypothesis_filepath)
            print_and_write(filename=log_filename, log='[*] Output Adaboost hypothesis to "{}" success.'.format(hypothesis_filepath))
        except Exception as e:
            print_and_write(filename=log_filename, log='[!] {}'.format(e))
            print_and_write(filename=log_filename, log='[!] Output Adaboost hypothesis failed.')
        print_and_write(filename=log_filename, log='-'*100)

    else:
        # Print and write label mapping
        print('-'*100)
        print('[*] Class label mapping:')
        print(' -  {} => +1'.format(data['+1']))
        print(' -  {} => -1'.format(data['-1']))

        # Train on full training data and evaluate on test data with the specified hyper-parameters
        param = {'num_classifer': args.T}
        model = AdaboostClassifer(data, param=param, verbose=True)
        model.train()

        # Compute training and test error.
        train_error = zero_one_loss(y_truth=data['train_y'], y_pred=model.hypothesis(X=data['train_X']))
        test_error = zero_one_loss(y_truth=data['test_y'], y_pred=model.hypothesis(X=data['test_X']))

        # Print results
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print('-'*100)
        print('[*] Datetime: {}'.format(timestamp))
        print('[*] Train file path: "{}"'.format(train_filepath))
        print('[*] Test file path: "{}"'.format(test_filepath))
        print('[*] Specified parameter: {}'.format(param))
        print('[*] Hypothesis:')
        print('{}'.format(model))
        print('[*] Performance: Train error: {} | Test error: {}'.format(train_error, test_error))
        print('-'*100)

        # Output Adaboost hypothesis (need training data with header)
        hypothesis_filepath = 'hypothesis/Adaboost_hypothesis_header-[{}].csv'.format(timestamp)
        print('[*] Saving Adaboost hypothesis to "{}" ...'.format(hypothesis_filepath))
        try:
            model.to_csv(filepath=hypothesis_filepath)
            print('[*] Output Adaboost hypothesis to "{}" success.'.format(hypothesis_filepath))
        except Exception as e:
            print('[!] {}'.format(e))
            print('[!] Output Adaboost hypothesis failed.')
        print('-'*100)
