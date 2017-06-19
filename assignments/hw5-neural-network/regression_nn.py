import numpy as np
import pandas as pd

from core import *

class RegressionNN:
    """ Neural network regression model wrapper for training and testing.
        Currently all activation functions are `Sigmoid`, instead of input layer
        and output layer are `Linear`.
    """
    def __init__(self, data, param={}, cost=SquaredError, regularizer=L2):

        # Assign data
        X = data['train_X'].astype('float')
        y = data['train_y'].astype('float')
        if y.ndim == 1: y = y[:,np.newaxis]

        # Shuffle
        X, y = self._shuffle(X, y)

        # Split training set and validation set.
        valid_size = X.shape[0]//3
        self.X_valid = X[:valid_size]
        self.y_valid = y[:valid_size]
        self.X_train = X[valid_size:]
        self.y_train = y[valid_size:]

        self.X_test = data['test_X'].astype('float')
        self.y_test = data['test_y'].astype('float')
        if self.y_test.ndim == 1: self.y_test = self.y_test[:,np.newaxis]

        # Hyperparameters
        self.input_size = self.X_train.shape[1]
        self.output_size = self.y_train.shape[1]
        self.depth = param.get('depth', 2)
        self.hidden_sizes = param.get('hidden_sizes', [20] * (self.depth-1)) # Default: [20]. Single hidden layer.
        self.learning_rate = param.get('learning_rate', 0.01)
        self.max_iteration = param.get('max_iteration', 30000)
        self.reg_lambda = param.get('reg_lambda', 0.0) # Weight regularization strength
        self.regularizer = regularizer(reg_lambda=self.reg_lambda) if issubclass(regularizer, Regularizer) else L2(reg_lambda=self.reg_lambda)
        self.cost = cost() if issubclass(cost, Cost) else SquaredError() # Default: squared error. Must support `delta()`.

        # Initialization
        self._init_model()
        self._init_weights()

    def _init_model(self):
        # Build layers
        layers = [Layer(self.input_size, Linear)]
        layers += [Layer(hidden_size, Sigmoid) for hidden_size in self.hidden_sizes]
        layers += [Layer(self.output_size, Linear)]
        self.network = Network(layers)

    def _init_weights(self):
        # Initialize weights with Guassian (0 mean, 1/sqrt(N) variance), were N the dimension of input space.
        self.weights = [np.random.normal(0, 1, shape) for shape in self.network.shapes]

    def _shuffle(self, X, y):
        # Shuffle examples.
        data = np.hstack([y, X])
        np.random.shuffle(data)
        X = data[:,1:]
        y = data[:,0]
        if y.ndim == 1: y = y[:,np.newaxis]
        return X, y

    def train(self, info=''):
        # Training with SGD.
        backprop = Backprop(network=self.network, cost=self.cost, regularizer=self.regularizer)
        update = GradientDecent()

        # Shuffle training set
        train_size = self.X_train.shape[0]
        X_train, y_train = self._shuffle(self.X_train, self.y_train)

        self.training = True
        self.best_valid_error = np.inf
        self.best_weights = self.weights
        self.patience = 500 # Number of iterations with no improvement after which training will be stopped.
        patience = 0

        for i in range(self.max_iteration):

            # Fetch training example
            example_idx = i % train_size
            x = X_train[example_idx]
            y = y_train[example_idx]

            # Update parameters.
            gradient = backprop(self.weights, x, y)
            self.weights = update(self.weights, gradient, learning_rate=self.learning_rate)

            # Monitor validation error.
            valid_error = self.evaluation(self.X_valid, self.y_valid)

            # Record best score and weights
            if valid_error < self.best_valid_error:
                self.best_valid_error = valid_error
                self.best_weights = self.weights
                patience = 0
            else:
                patience += 1

            if patience == self.patience:
                self.training = False
                print('-'*100)
                if info: print('[*] {}'.format(info))
                print('[*] Early stopping at {}-th iteration | Best valid error: {}'.format(i, self.best_valid_error))
                break

            if (i+1) == 1 or (i+1) % 100 == 0 or (i+1) == self.max_iteration:
                train_error = self.evaluation(self.X_train, self.y_train) # Just for logging
                print('-'*100)
                if info: print('[*] {}'.format(info))
                print('[*] Iteration: ({}/{}) | Patience: ({}/{}) | Train error: {} | Valid error: {}'.format(i+1, self.max_iteration, patience, self.patience, train_error, valid_error))

            if (i+1) == self.max_iteration:
                self.training = False
                print('-'*100)
                print('[*] Max iteration acheived. | Best valid error: {}'.format(self.best_valid_error))
                break

            # Shuffle training set after one epoch training.
            if (i+1) % train_size == 0:
                X_train, y_train = self._shuffle(self.X_train, self.y_train)

    def hypothesis(self, X):

        weights = self.weights if self.training else self.best_weights

        # Make predictions on single example or multiple examples
        if X.ndim == 1:
            return self.network.feed(weights, X)
        elif X.ndim == 2:
            return np.array([self.network.feed(weights, x) for x in X])

    def evaluation(self, X, y):
        # Default: Mean squared error + reg_lambda * regularization error.
        data_error = 0
        for i, (x, y) in enumerate(zip(X, y)):
            y_pred = self.hypothesis(x)
            data_error += self.cost(y, y_pred)
        data_error /= X.shape[0]

        if self.training:
            reg_error = self.regularizer(self.weights)
        else:
            reg_error = self.regularizer(self.best_weights)

        return data_error + reg_error

    def to_csv(self, filepath='hypothesis/SGD_hypothesis_header.csv'):
        df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([['depth', self.depth]])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame([['sizes'] + [self.input_size+1] \
                                                    + [hidden_size+1 for hidden_size in self.hidden_sizes] \
                                                    + [self.output_size]])], ignore_index=True)
        for i, weight in enumerate(self.best_weights):
            df = pd.concat([df, pd.DataFrame([['W_{}'.format(i)] + weight.T.flatten().tolist()])], ignore_index=True)

        # Fill nan with None[]
        df = df.where((pd.notnull(df)), None)

        # Since pd.to_csv converts int to float if there's `None` in the same row,
        # we need to handle this.
        with open(filepath, 'w') as f:
            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    if (row == 0 and col != 0) or (row == 1 and col != 0):
                        val = int(df[col][row]) if df[col][row] is not None else ''
                    else:
                        val = df[col][row] if df[col][row] is not None else ''
                    f.writelines('{},'.format(val))
                if row != df.shape[0]-1: f.writelines('\n')
