import pickle
import numpy as np

def normalize_binary_class_label(y):
    """ Normalize class label to {1, +1} if classes are different (e.g. {0, 1}, {'+', '-'}...) """

    binary_class_label = set(y)
    if binary_class_label == {1, -1}: return y
    if binary_class_label == {1, 0}: return np.vectorize(lambda y: 1 if y == 1 else -1)(y)
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

def split_k_fold(train_X, train_y, K, shuffle=False):
    train_test_folds = []
    for k in range(K):
        k_train_X = np.array([x for i, x in enumerate(train_X) if i % K != k])
        k_train_y = np.array([y for i, y in enumerate(train_y) if i % K != k])
        k_valid_X = np.array([x for i, x in enumerate(train_X) if i % K == k])
        k_valid_y = np.array([y for i, y in enumerate(train_y) if i % K == k])
        train_test_folds.append({ 'train_X': k_train_X, 'train_y': k_train_y, 'test_X': k_valid_X, 'test_y': k_valid_y })
    return train_test_folds

def k_fold_cross_validation(K, data, model, param):
    # Get K folds train, validation data.
    train_test_folds = split_k_fold(data['train_X'], data['train_y'], K=K)

    # Perform cross validation
    cv_error = 0

    for i, data in enumerate(train_test_folds):

        # Train SVM
        _model = model(data=data, param=param)
        _model.train(info='{}-th fold | parameter: {}'.format(i+1, param))

        # Compute training and test error.
        train_error, test_error = _model.compute_error()

        print('[*] {}-th fold | train error: {} | test error: {}'.format(i+1, train_error, test_error))

        cv_error += test_error

    cv_error /= K

    return _model, cv_error

def print_and_write(filename='log.txt', log=''):
    print(log)
    with open(filename, 'a') as log_file:
        log_file.write(log + '\n')
