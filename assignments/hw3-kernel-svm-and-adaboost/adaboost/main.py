import datetime
import argparse
import numpy as np

from adaboost import AdaboostClassifer
from utils import *

parser = argparse.ArgumentParser(description='Binary Adaboost classifer.')
parser.add_argument('--train_filename', dest='train_filename', type=str, default='messidor_features_training.csv', help='Training dataset csv. (Default: "messidor_features_training.csv")')
parser.add_argument('--test_filename', dest='test_filename', type=str, default='messidor_features_testing.csv', help='Training dataset csv. (Default: "messidor_features_testing.csv")')
parser.add_argument('--K', dest='K', type=int, default=None, help='Denotes for "K"-fold cross-validation for determine the optimal hyper-parameters for adaboost classifer. (Default: None)')
parser.add_argument('--T', dest='T', type=int, default=5, help='The maximum number of classifers at which boosting is terminated.')

args = parser.parse_args()

if __name__ == '__main__':

    # Load dataset.
    data = load_dataset(train_file_path='dataset/' + args.train_filename,
                        test_file_path='dataset/' + args.test_filename)

    # Perform grid search K-fold cross validation.
    if args.K:
        log_filename = 'adaboost-{}-fold-[{}].txt'.format(args.K, datetime.datetime.now().strftime('%H:%M:%S'))
        
        # Specify list of hyper-parameters for grid search K-fold cross validation.
        param_grid = {'num_classifer': [1, 2 ,3, 4, 5]}

        cv = GridSearchCV(data=data, model=AdaboostClassifer, param_grid=param_grid, num_folds=args.K)
        cv.train()

        # Print and write CV results to log file.
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Cross validation history:')
        for cv_result in cv.cv_results:
            print_and_write(filename=log_filename, log=' - Parameter: {} | Cross validation error: {}'.format(cv_result['param'], cv_result['score']))
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
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Datetime: {}'.format(datetime.datetime.now().strftime('%H:%M:%S')))
        print_and_write(filename=log_filename, log='[*] Train filename: {}'.format(args.train_filename))
        print_and_write(filename=log_filename, log='[*] Test filename: {}'.format(args.test_filename))
        print_and_write(filename=log_filename, log='[*] Best parameter: {}'.format(cv.best_param))
        print_and_write(filename=log_filename, log='[*] Performance: Train error: {} | Test error: {}'.format(train_error, test_error))
        print_and_write(filename=log_filename, log='-'*100)

    else:
        # Train on full training data and evaluate on test data with the specified hyper-parameters
        param = {'num_classifer': args.T}
        model = AdaboostClassifer(data, param=param, verbose=True)
        model.train()

        # Compute training and test error.
        train_error = zero_one_loss(y_truth=data['train_y'], y_pred=model.hypothesis(X=data['train_X']))
        test_error = zero_one_loss(y_truth=data['test_y'], y_pred=model.hypothesis(X=data['test_X']))

        # Print results
        print('-'*100)
        print('[*] Datetime: {}'.format(datetime.datetime.now().strftime('%H:%M:%S')))
        print('[*] Train filename: {}'.format(args.train_filename))
        print('[*] Test filename: {}'.format(args.test_filename))
        print('[*] Specified parameter: {}'.format(param))
        print('[*] Performance: Train error: {} | Test error: {}'.format(train_error, test_error))
        print('-'*100)
