""" Binary SVM optimized with SMO algorithm
    Reference:
        - SMO supplement provided by EE6550 course
        - Standford CS 229, Autumn 2009 The Simplified SMO Algorithm
        - Improvment to Platt's SMO Algorithm for SVM Classifer Design, S.S. Keerthi et al
        - A Roadmap to SVM Sequential Minimal Optimization for Classification by Ruben Ramirez-Padron

    Author: Howard (Yu-Chun) Lo
"""

import numpy as np

class SVM:
    def __init__(self, data, param, kernel_type='linear'):
        # Upack training and test data
        # X: shape(N, D), where N is the number of examples, D is data dimension
        # Y: shape(N,)
        self.train_X = data['train_X']
        self.train_y = data['train_y']
        self.test_X = data['test_X']
        self.test_y = data['test_y']

        # Unpack hyper-parameters
        self.C = param['C'] # Regularization term
        self.tol = 1e-3 # Numerical tolerance for checking KKT conditions

        # Hyper-parameters
        self.kernels = {
            'linear': lambda x_i, x_j: np.dot(x_i, x_j)
        }
        self.kernel = self.kernels[kernel_type]

        # Model parameters
        self.w = np.zeros(self.train_X.shape[1]) # Weight vector: shape(D,)
        self.b = 0.0 # Bias term: scalar
        self.alpha = np.zeros(len(self.train_X)) # Lagrange multipliers

        # Max number of iteration to avoid not converging
        self.max_iteration = 100000

    def train(self, info=''):
        """ Optimize alpha with method combining the simple SMO algorithm and Platt's algorithm
            In each iteration, the SMO algorithm solves the Lagrangian dual problem
            which involves only two Lagrangian multipliers.
        """

        num_changed_alphas = 0
        examine_all = 1
        iteration = 0

        while num_changed_alphas > 0 or examine_all:
            num_changed_alphas = 0
            if examine_all:
                # Repeated pass iterates over entire examples.
                for i in range(len(self.train_X)):
                    # alpha_i needs update, select alpha_j (!= alpha_i) to jointly optimize the alpha pair
                    if self._violate_KKT_conditions(i):
                        j = i
                        while(j == i): j = np.random.randint(0, len(self.train_X))
                        # Update alpha_i and alpha_j
                        num_changed_alphas += self._update_alpha_pair(i, j)

                print('-'*100)
                print('[*] {} | One passes: {} alphas changed.'.format(info, num_changed_alphas))
                
                if num_changed_alphas == 0:
                    print('[*] {} | Converged.'.format(info))
                    print('-'*100)
                else:
                    print('[*] {} | Go to repeated passes.'.format(info))
            else:
                # Repeated pass iterates over non-boundary examples.
                I_non_boundary = np.where(np.logical_and(self.alpha > 0, self.alpha < self.C) == True)[0].tolist()
                if len(I_non_boundary):
                    E_list = np.vectorize(self._E)(I_non_boundary)
                    if not max(E_list) - min(E_list) < 1:
                        for i in I_non_boundary:
                            num_changed_alphas += self._examine_example(i)

                if num_changed_alphas == 0:
                    print('-'*100)
                    print('[*] {} | Repeated passes done. Go back to one pass.'.format(info))

            if examine_all == 1:
                # One pass done, go to repeated passes.
                examine_all = 0
            elif num_changed_alphas == 0:
                # Repeated pass done, go back to one pass.
                examine_all = 1

            iteration += 1
            if iteration == 1 or iteration % 1000 == 0 or iteration == self.max_iteration:
                train_error, test_error = self.compute_error()
                print('-'*100)
                print('[*] {} | iteration: {} | train error: {} | test error: {}'.format(info, iteration, train_error, test_error))

            if iteration == self.max_iteration:
                print('-'*100)
                print('[*] Max iteration acheived.')
                return

    def hypothesis(self, X):
        """ Applying our linear classifier `f(x)` to perform binary classification.
            If f(x) >= 0, y(i) = +1
            Else    <  0, y(i) = -1

            @param `X`: X can be a single example with shape(D,) or multiple examples with shape(N, D)
        """
        return np.sign(self._f(X))

    def compute_error(self):
        """ Compute training error and test error """

        y_pred = self.hypothesis(self.train_X)
        train_error = np.mean(y_pred != self.train_y)

        y_pred = self.hypothesis(self.test_X)
        test_error = np.mean(y_pred != self.test_y)

        return train_error, test_error

    def _violate_KKT_conditions(self, i):
        """ Check if an example violates the KKT conditons """
        alpha_i = self.alpha[i]
        R_i = self.train_y[i]*self._E(i)

        return (R_i < -self.tol and alpha_i < self.C) or (R_i > self.tol and alpha_i > 0)

    def _examine_example(self, i):
        """ Implement Platt's heuristics to select a good alpha pair to optimize.
            (First heuristic is not implemented since it makes training slower)
        """
        # Check if alpha_i needs updating (alpha_i violates KKT conditions)
        if self._violate_KKT_conditions(i):

            # Retrieve indexes of non boundary examples
            I_non_boundary = np.where(np.logical_and(self.alpha > 0, self.alpha < self.C) == True)[0].tolist()

            # Iterate over non-boundary items, starting at a random position
            shuffled_I_non_boundary = np.copy(I_non_boundary)
            np.random.shuffle(shuffled_I_non_boundary)
            for j in shuffled_I_non_boundary:
                if self._update_alpha_pair(i, j):
                    return 1

            # Iterate over entire items, starting at a random position
            I = np.arange(len(self.train_X))
            shuffled_I = np.copy(I)
            np.random.shuffle(shuffled_I)
            for j in shuffled_I:
                if self._update_alpha_pair(i, j):
                    return 1
        return 0

    def _update_alpha_pair(self, i, j):
        """ Jointly optimized alpha_i and alpha_j """
        # Not the alpha pair.
        if i == j: return 0

        E_i = self._E(i)
        E_j = self._E(j)

        alpha_i = self.alpha[i]
        alpha_j = self.alpha[j]

        x_i, x_j, y_i, y_j = self.train_X[i], self.train_X[j], self.train_y[i], self.train_y[j]

        if y_i == y_j:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)
        else:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)

        # This will not make any progress.
        if L == H: return 0

        # Compute eta (second derivative of the Lagrange dual function = -eta)
        eta = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - 2*self.kernel(x_i, x_j)

        # eta > 0 => second derivative(-eta) < 0 => maximum exists.
        if eta <= 0: return 0

        # Compute new alpha_j and clip it inside [L, H]
        alpha_j_new = alpha_j + y_j*(E_i - E_j)/eta
        if alpha_j_new < L: alpha_j_new = L
        if alpha_j_new > H: alpha_j_new = H

        # Compute new alpha_i based on new alpha_j
        alpha_i_new = alpha_i + y_i*y_j*(alpha_j - alpha_j_new)

        # Compute step sizes
        delta_alpha_i = alpha_i_new - alpha_i
        delta_alpha_j = alpha_j_new - alpha_j

        # Step size too small to update
        if abs(delta_alpha_j) < 1e-5: return 0

        # Update weight vector
        self.w = self.w + delta_alpha_i*y_i*x_i + delta_alpha_j*y_j*x_j

        # Update b
        b_i = self.b - E_i - delta_alpha_i*y_i*self.kernel(x_i, x_i) - delta_alpha_j*y_j*self.kernel(x_i, x_j)
        b_j = self.b - E_j - delta_alpha_i*y_i*self.kernel(x_i, x_j) - delta_alpha_j*y_j*self.kernel(x_j, x_j)
        self.b = (b_i + b_j)/2
        if (alpha_i_new > 0 and alpha_i_new < self.C):
            self.b = b_i
        if (alpha_j_new > 0 and alpha_j_new < self.C):
            self.b = b_j

        # Update the alpha pair
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new

        return 1

    def _f(self, X):
        """ Linear classifier `f(x)`, used when training or making predictions.

            @param `X`: `X` can be a single example with shape(D,) or multiple examples with shape(N, D)
        """
        return np.dot(X, self.w) + self.b

    def _E(self, i):
        """ Prediction error: _f(x_i) - y_i, used when training. """
        return self._f(self.train_X[i]) - self.train_y[i]
