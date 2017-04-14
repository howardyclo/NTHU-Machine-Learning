import pickle
import numpy as np

from itertools import product

def normalize_binary_class_label(y):
    """ Normalize class label to {1, +1} if classes are different (e.g. {0, 1}, {'+', '-'}...) """

    binary_class_label = set(y)
    if binary_class_label == {1, -1}: return y
    if binary_class_label == {1, 0}: return np.vectorize(lambda y: 1 if y == 1 else -1)(y)
    if binary_class_label == {1, 2}: return np.vectorize(lambda y: 1 if y == 1 else -1)(y)
    if binary_class_label == {'+', '-'}: return np.vectorize(lambda y: 1 if y == '+' else -1 )(y)
    return np.vectorize(lambda y: 1 if y == list(binary_class_label)[0] else -1 )(y)

def load_dataset(train_file_path='dataset/messidor_features_training.csv', test_file_path='dataset/messidor_features_testing.csv'):
    """ Load training, test dataset only in csv """

    train_data = np.genfromtxt(train_file_path, delimiter=',')
    train_X = train_data[:,1:]
    train_y = train_data[:,0]

    test_data = np.genfromtxt(test_file_path, delimiter=',')
    test_X = test_data[:,1:]
    test_y = test_data[:,0]

    train_test_y = train_y.tolist() + test_y.tolist() # concatenate
    train_test_y = normalize_binary_class_label(train_test_y) # normalize binary label to {+1, -1}
    train_y = train_test_y[:len(train_y)] # split to train_y
    test_y = train_test_y[len(train_y):] # split to test_y

    return { 'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y }

def print_and_write(filename='log.txt', log=''):
    print(log)
    with open(filename, 'a') as log_file:
        log_file.write(log + '\n')

def zero_one_loss(y_truth, y_pred, sample_weights=None):
    return np.average(y_pred != y_truth, weights=sample_weights)

def split_K_fold(X, y, K, shuffle=False):
    train_test_folds = []

    for k in range(K):
        k_train_X = np.array([_x for i, _x in enumerate(X) if i % K != k])
        k_train_y = np.array([_y for i, _y in enumerate(y) if i % K != k])
        k_valid_X = np.array([_x for i, _x in enumerate(X) if i % K == k])
        k_valid_y = np.array([_y for i, _y in enumerate(y) if i % K == k])

        train_test_folds.append({'train_X': k_train_X,
                                 'train_y': k_train_y,
                                 'test_X': k_valid_X,
                                 'test_y': k_valid_y})
    return train_test_folds

def K_fold_cross_validation(data, model, param, scorer=zero_one_loss, K=5):
    # Unpack data
    train_X = data['train_X']
    train_y = data['train_y']
    test_X = data['test_X']
    test_y = data['test_y']

    # Get K folds training, validation data.
    train_test_folds = split_K_fold(X=train_X, y=train_y, K=K)

    # Perform cross-validation.
    cv_error = 0
    for i, data in enumerate(train_test_folds):

        _model = model(data=data, param=param)
        _model.train()

        # Compute training and test error.
        train_error = scorer(y_truth=train_y, y_pred=_model.hypothesis(X=train_X))
        test_error = scorer(y_truth=test_y, y_pred=_model.hypothesis(X=test_X))

        print('-'*100)
        print('[*] {}-th fold | Parameter: {}'.format(i+1, param))
        print('[*] {}-th fold | Train error: {} | Validation error: {}'.format(i+1, train_error, test_error))

        cv_error += test_error

    cv_error /= K

    return cv_error

class GridSearchCV:
    def __init__(self, data, model=None, param_grid={}, scorer=zero_one_loss, num_folds=5):
        """ Exhaustive search over specified parameter values for a model.

            @param data: Dictionary of training and test data:
                - data['train_X']: Training data with shape(N, D), where N is the number of examples, D is data dimension.
                - data['train_y']: Training label with shape(N,)
                - data['test_X']: Test data with shape(N, D), where N is the number of examples, D is data dimension.
                - data['test_y']: Test label with shape(N,)

            @param `model`: Either classifier or regression class that supports model.hypothesis(X) for evaluation.
            @param `param_grid`: Dictionary with parameters names (string) as keys and lists of parameter settings as values.
            @param `scorer`: Model evaluation function. (Defualt: zero_one_loss)
            @param `num_folds`: For K-fold cross-validation. (Defualt: 5)
        """
        self.data = data
        self.model = model
        self.scorer = scorer
        self.num_folds = num_folds
        self.params = self._span_param_grid(param_grid)

        self.cv_results = []
        self.best_param = {}
        self.best_score = 0.0

    def train(self):
        """ Find the best hyper-parameters with K-fold cross-validation. """
        for param in self.params:
            score = K_fold_cross_validation(data=self.data,
                                            model=self.model,
                                            param=param,
                                            scorer=self.scorer,
                                            K=self.num_folds)
            self.cv_results.append({'param': param, 'score': score})

        self.cv_results.sort(key=lambda item: item['score'])
        self.best_score = self.cv_results[0]['score']
        self.best_param = self.cv_results[0]['param']

    def _span_param_grid(self, param_grid):
        """ Perform Cartesian prodcut on `self.param_grid`
            - Input: param_grid = {'a': [1, 2], 'b': [True, False]}
            - Output: [{'a': 1, 'b': True}, {'a': 1, 'b': False},
                       {'a': 2, 'b': True}, {'a': 2, 'b': False}]
        """
        return list(dict(zip(param_grid, x)) for x in product(*param_grid.values()))
