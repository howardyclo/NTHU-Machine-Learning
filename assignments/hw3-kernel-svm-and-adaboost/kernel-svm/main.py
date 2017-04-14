import sys
import datetime
import argparse
import itertools
import numpy as np

from utils import *
from svm import SVM

parser = argparse.ArgumentParser(description='Binary support vector classifer.')
parser.add_argument('--train_filename', dest='train_filename', type=str, default='messidor_features_training.csv', help='Training dataset csv. (Default: "messidor_features_training.csv")')
parser.add_argument('--test_filename', dest='test_filename', type=str, default='messidor_features_testing.csv', help='Training dataset csv. (Default: "messidor_features_testing.csv")')
parser.add_argument('--K', dest='K', type=int, default=None, help='Denotes for "K"-fold cross-validation for determine the optimal hyper-parameters for SVM. (Default: None)')
parser.add_argument('--C', dest='C', type=float, default=0.1, help='Parameter for penalty term. (Default: 0.1)')
parser.add_argument('--kernel_type', dest='kernel_type', type=str, default=None, help='Kernel type to be used in SVM. Acceptable kernel type: "linear", "poly", "rbf". (Default: None)')
parser.add_argument('--poly_degree', dest='poly_degree', type=int, default=3, help='Degree of the polynomial kernel function ("poly"). Ignored by all other kernels. (Default: 3)')
parser.add_argument('--rbf_sigma', dest='rbf_sigma', type=float, default=0.5, help='Sigma term in RBF (guassian). Ignored by all other kernels. (Default: 0.5)')
parser.add_argument('--enable_heuristic', dest='enable_heuristic', action='store_true', default=False, help='Whether use Platts heuristics to train SVM. (Defualt: False)')
parser.add_argument('--enable_kernel_cache', dest='enable_kernel_cache', action='store_true', default=True, help='Whether precompute kernel results. This can speed up training but need time to initialize when data is large. (Defualt: True)')
parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=20000, help='Max iteration for SMO training algorithm to avoid not converging.')

args = parser.parse_args()

if __name__ == '__main__':

    # Load dataset.
    data = load_dataset(train_file_path='dataset/' + args.train_filename,
                        test_file_path='dataset/' + args.test_filename)

    # Perform grid search K-fold cross validation.
    if args.K:
        if args.kernel_type is None:
            print('[!] Please specify argument --kernel_type for K-fold cross-validation.')
            print('[!] Acceptable kernel type: "linear", "poly", "rbf". (Default: None)')
            print('[!] Exit program.')
            sys.exit(0)

        log_filename = 'svm-{}-{}-fold-[{}].txt'.format(args.kernel_type, args.K, datetime.datetime.now().strftime('%H:%M:%S'))

        # Specify list of hyper-parameters for grid search K-fold cross validation.
        param_grid = {
            'C': np.linspace(0.1, 1.0, 15).round(2).tolist(),
            'kernel_type': [args.kernel_type]
        }
        if args.kernel_type == 'poly':
            param_grid['poly_degree'] = [2, 3, 4, 5]
        elif args.kernel_type == 'rbf':
            param_grid['rbf_sigma'] = np.linspace(0.1, 1.0, 5).round(2).tolist()

        cv = GridSearchCV(data=data, model=SVM, param_grid=param_grid, num_folds=args.K)
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
        model = SVM(data=data, param=cv.best_param, verbose=True)
        model.train()

        # Compute training and test error.
        train_error = zero_one_loss(y_truth=data['train_y'], y_pred=model.hypothesis(X=data['train_X']))
        test_error = zero_one_loss(y_truth=data['test_y'], y_pred=model.hypothesis(X=data['test_X']))

        # Print and write results to log file.
        print_and_write(filename=log_filename, log='-'*100)
        print_and_write(filename=log_filename, log='[*] Train filename: {}'.format(args.train_filename))
        print_and_write(filename=log_filename, log='[*] Test filename: {}'.format(args.test_filename))
        print_and_write(filename=log_filename, log='[*] Datetime: {}'.format(datetime.datetime.now().strftime('%H:%M:%S')))
        print_and_write(filename=log_filename, log='[*] Best parameter: {}'.format(cv.best_param))
        print_and_write(filename=log_filename, log='[*] Lagrange multipliers: {}'.format(model.alpha))
        if model.use_w: print_and_write(filename=log_filename, log='[*] Weight vector: {}'.format(model.w))
        print_and_write(filename=log_filename, log='[*] Bias term: {}'.format(model.b))
        print_and_write(filename=log_filename, log='[*] Performance: Train error: {} | Test error: {}'.format(train_error, test_error))
        print_and_write(filename=log_filename, log='-'*100)
    else:
        if args.kernel_type is None:
            print('[!] Please specify argument --kernel_type.')
            print('[!] Acceptable kernel type: "linear", "poly", "rbf". (Default: None)')
            print('[!] Exit program.')
            sys.exit(0)

        # Specify hyper-parameters
        param = { 'C': args.C, 'kernel_type': args.kernel_type }
        if args.kernel_type == 'poly':
            param['poly_degree'] = args.poly_degree
        elif args.kernel_type == 'rbf':
            param['rbf_sigma'] = args.rbf_sigma

        # Train on full training data and evaluate on test data with the specified hyper-parameters
        model = SVM(data=data, param=param)
        model.train(info='Train on specified parameter: {}'.format(param))

        # Compute training and test error.
        train_error = zero_one_loss(y_truth=data['train_y'], y_pred=model.hypothesis(X=data['train_X']))
        test_error = zero_one_loss(y_truth=data['test_y'], y_pred=model.hypothesis(X=data['test_X']))

        # Print results
        print('-'*100)
        print('[*] Datetime: {}'.format(datetime.datetime.now().strftime('%H:%M:%S')))
        print('[*] Train filename: {}'.format(args.train_filename))
        print('[*] Test filename: {}'.format(args.test_filename))
        print('[*] Specified parameter: {}'.format(param))
        print('[*] Lagrange multipliers: {}'.format(model.alpha))
        if model.use_w: print('[*] Weight vector: {}'.format(model.w))
        print('[*] Bias term: {}'.format(model.b))
        print('[*] Performance: Train error: {} | Test error: {}'.format(train_error, test_error))
        print('-'*100)
