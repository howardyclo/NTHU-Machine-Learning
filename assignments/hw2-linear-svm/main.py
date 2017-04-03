import datetime
import argparse
import numpy as np

from utils import *
from svm import SVM

parser = argparse.ArgumentParser(description='Linear support vector classifer.')
parser.add_argument('--train_filename', dest='train_filename', type=str, default='messidor_features_training.csv', help='Training dataset csv. (default: "messidor_features_training.csv")')
parser.add_argument('--test_filename', dest='test_filename', type=str, default='messidor_features_testing.csv', help='Training dataset csv. (default: "messidor_features_testing.csv")')
parser.add_argument('--K', dest='K', type=int, default=5, help='Denotes for "K"-fold cross-validation for determine the optimal value C for SVM. (default: 5)')
parser.add_argument('--C', dest='C', type=float, default=None, help='If C is specified, disable cross-validation, train SVM on this specified C (default: None)')

args = parser.parse_args()

if __name__ == '__main__':

    # Load dataset
    data = load_dataset(train_file_path='dataset/' + args.train_filename,
                        test_file_path='dataset/' + args.test_filename)

    if not args.C:
        # Perform hyper-parameter optimization (finding best `C` regularization term)
        num_split = 5
        C_list = np.linspace(0.1, 1.0, num_split).round(2).tolist()
        log_filename = '{}-fold-result-[{}].txt'.format(args.K, datetime.datetime.now().strftime('%H:%M:%S'))

        depth = 0
        max_depth = 2
        cv_history = []

        while depth < max_depth:

            for C in C_list:
                param = {'C': C}
                _model, cv_error = k_fold_cross_validation(K=args.K, data=data, model=SVM, param=param)
                cv_history.append({'param': param, 'cv_error': cv_error})

                print_and_write(filename=log_filename, log='*'*100)
                print_and_write(filename=log_filename, log='* Datetime: {}'.format(datetime.datetime.now().strftime('%H:%M:%S')))
                print_and_write(filename=log_filename, log='* Parameter: {} | cross validation error: {}'.format(param, cv_error))
                print_and_write(filename=log_filename, log='*'*100)

            cv_history.sort(key=lambda item: item['cv_error'])
            C_list = np.linspace(cv_history[0]['param']['C'], cv_history[1]['param']['C'], num_split).round(2).tolist()
            depth += 1

        cv_history.sort(key=lambda item: item['cv_error'])
        best_param = cv_history[0]['param']
        best_cv_error = cv_history[0]['cv_error']

        print_and_write(filename=log_filename, log='*'*100)
        print_and_write(filename=log_filename, log='* Datetime: {}'.format(datetime.datetime.now().strftime('%H:%M:%S')))
        print_and_write(filename=log_filename, log='* Best parameter: {} | cross validation error: {}'.format(best_param, best_cv_error))
        print_and_write(filename=log_filename, log='* Start to train on full training data and evaluate on test data.')
        print_and_write(filename=log_filename, log='*'*100)

        # Train on full training data and evaluate on test data with the best hyper-parameters
        model = SVM(data=data, param=best_param)
        model.train(info='Train on best parameter: {}'.format(best_param))
        train_error, test_error = model.compute_error()

        print_and_write(filename=log_filename, log='*'*100)
        print_and_write(filename=log_filename, log='* Datetime: {}'.format(datetime.datetime.now().strftime('%H:%M:%S')))
        print_and_write(filename=log_filename, log='* Optimal parameter: {}'.format(best_param))
        print_and_write(filename=log_filename, log='* Weight vector: {}'.format(model.w))
        print_and_write(filename=log_filename, log='* Bias term: {}'.format(model.b))
        print_and_write(filename=log_filename, log='* Performance: train error: {} | test error: {}'.format(train_error, test_error))
        print_and_write(filename=log_filename, log='*'*100)
    else:
        # Train on full training data and evaluate on test data with the specified hyper-parameters
        param = {'C': args.C}
        model = SVM(data=data, param=param)
        model.train(info='Train on specified parameter: {}'.format(param))
        train_error, test_error = model.compute_error()

        print('*'*100)
        print('* Datetime: {}'.format(datetime.datetime.now().strftime('%H:%M:%S')))
        print('* Specified parameter: {}'.format(param))
        print('* Weight vector: {}'.format(model.w))
        print('* Bias term: {}'.format(model.b))
        print('* Performance: train error: {} | test error: {}'.format(train_error, test_error))
        print('*'*100)
