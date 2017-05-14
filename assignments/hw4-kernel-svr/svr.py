""" Author: Howard (Yu-Chun) Lo """

import numpy as np

from utils import MeanAbsoluteEpsilonError

class SVR:
    def __init__(self, data, param={}, scorer=None, enable_heuristic=False, enable_kernel_cache=True, max_iteration=3000, verbose=False):
        """ Support vector regression optimized with SMO algorithm

            Implementation Reference:
                - SMO supplement provided by EE6550 course
                - Standford CS 229, Autumn 2009 The Simplified SMO Algorithm
                - Improvment to Platt's SMO Algorithm for SVM Classifer Design, S.S. Keerthi et al
                - A Roadmap to SVM Sequential Minimal Optimization for Classification by Ruben Ramirez-Padron

            @param data: Dictionary of training and test data:
                - data['train_X']: Training data with shape(N, D), where N is the number of examples, D is data dimension.
                - data['train_y']: Training label with shape(N,)
                - data['test_X']: Test data with shape(N, D), where N is the number of examples, D is data dimension.
                - data['test_y']: Test label with shape(N,)

            @param param: Dictionary of hyper-parameters:
                - param['C']: Parameter for penalty term. (Default: 0.1)
                - param['epsilon']: Epsilon in the epsilon-SVR model (Default: 0.1)
                - param['tol']: Tolerance for KKT conditons. (Defualt: 1e-2)
                - param['kernel_type']: Kernel type to be used in SVR. Acceptable kernel type: 'linear', 'poly', 'rbf'. (Default: 'linear')
                - param['poly_degree']: Degree of the polynomial kernel function ('poly'). Ignored by all other kernels. (Default: 3)
                - param['rbf_sigma']: Sigma term in RBF (guassian). Ignored by all other kernels. (Default: 0.5)

            @param `scorer`: For monitoring error during training. (Default: utils.MeanAbsoluteEpsilonError(epsilon=param['epsilon']))
            @param `enable_heuristic`: Whether use Platts heuristics to train SVR. (Defualt: False)
            @param `enable_kernel_cache`: Whether precompute kernel results. This can speed up training but need time to initialize when data is large. (Defualt: True)
            @param `max_iteration`: Max iteration for SMO training algorithm to avoid not converging. (Default: 5000)
        """
        # Upack training and test data
        self.train_X = data['train_X']
        self.train_y = data['train_y']
        self.test_X = data['test_X']
        self.test_y = data['test_y']

        # Unpack hyper-parameters
        self.C = param.get('C', 0.1)
        self.tol = param.get('tol', 1e-2)
        self.epsilon = param.get('epsilon', 1e-1)
        self.kernel_type = param.get('kernel_type', 'linear')
        self.poly_degree = param.get('poly_degree', 3)
        self.rbf_sigma = param.get('rbf_sigma', 0.5)

        self.scorer = scorer if scorer is not None else MeanAbsoluteEpsilonError(epsilon=self.epsilon)
        self.enable_heuristic = enable_heuristic
        self.enable_kernel_cache = enable_kernel_cache
        self.max_iteration = max_iteration
        self.verbose = verbose

        # Set kernel function
        self.kernels = {
            'linear': self._linear_kernel,
            'poly': self._poly_kernel,
            'rbf': self._rbf_kernel
        }
        self.kernel = self.kernels[self.kernel_type]

        # Precompute kernel cache
        if self.enable_kernel_cache:
            print('-'*100)
            print('[*] Enable kernel cache. Precomputing kernel results for all training examples ...')
            self.kernel_cache = self._precompute_kernel_cache()

        # Model parameters
        self.use_w = True if self.kernel_type == 'linear' else False # If linear kernel is used, use weight to perform prediction instead of alphas.
        self.w = np.zeros(self.train_X.shape[1]) # Weight vector: shape(D,) this will be updated when training.
        self.b = 0.0 # Bias term: scalar, this will be updated when training.

        # SVR needs a pair of lagrange multipliers
        # This alpha is not the same as SVC, where each example has one lagrange multiplier `alpha`.
        # In SVR, each example has two lagrange multipliers `a1` and `a2`.
        # Here, alpha is actually ( a2 - a1 ).
        self.alpha = np.zeros(len(self.train_X))

        # After training, we can compute biases for support vectors (training examples which 0 < alpha < C)
        # for estimating sample mean and sample std of biases.
        # For a good learning result, sample std of biases should be small.
        self.postcomputed_biases = np.array([None]*len(self.train_X))
        self.b_mean = None # Instead of using self.b to make prediction, use self.b_mean when training is done.
        self.b_std = None

    def train(self, info=''):
        """ Optimize alpha with either simple SMO algorithm or simple SMO combined with Platt's heuristics
            In each iteration, the SMO algorithm solves the Lagrangian dual problem
            which involves only two Lagrangian multipliers.
        """
        if self.enable_heuristic:
            self._heuristic_smo(info)
        else:
            self._simple_smo(info)

    def hypothesis(self, X):
        """ Applying our linear classifier `f(x)` to perform binary classification.
            If f(x) >= 0, y(i) = +1
            Else    <  0, y(i) = -1

            @param `X`: X can be a single example with shape(D,) or multiple examples with shape(N, D)
        """
        # ---- Not the same as SVC ----
        return self._f(X)

    def _simple_smo(self, info=''):

        num_changed_alphas = 1
        iteration = 0

        while num_changed_alphas > 0:
            num_changed_alphas = 0
            for i in range(len(self.train_X)):
                if self._violate_KKT_conditions(i):
                    j = i
                    while(j == i): j = np.random.randint(0, len(self.train_X))
                    num_changed_alphas += self._update_alpha_pair(i, j)

            if self.verbose and num_changed_alphas == 0:
                if info: print('[*] {}'.format(info))
                print('[*] Converged at iteration {}.'.format(iteration+1))
                print('-'*100)

            iteration += 1
            if self.verbose and (iteration == 1 or iteration % 100 == 0 or iteration == self.max_iteration):
                # Compute training and testing error
                train_error = self.scorer(y_truth=self.train_y, y_pred=self.hypothesis(X=self.train_X))
                test_error = self.scorer(y_truth=self.test_y, y_pred=self.hypothesis(X=self.test_X))
                print('-'*100)
                if info: print('[*] {}'.format(info))
                print('[*] {} alphas changed.'.format(num_changed_alphas))
                print('[*] Iteration: {} | Train error: {} | Test error: {}'.format(iteration, train_error, test_error))

            if iteration == self.max_iteration:
                print('-'*100)
                print('[*] Max iteration acheived.')
                break

        if self.verbose: print('[*] Averaging post-computed biases as final bias of SVR hypothesis.')
        self._postcompute_biases()

    def _heuristic_smo(self, info=''):

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

                if self.verbose:
                    print('-'*100)
                    if info: print('[*] {}'.format(info))
                    print('[*] One passes done.')

                if self.verbose and num_changed_alphas == 0:
                    if info: print('[*] {}'.format(info))
                    print('[*] Converged at iteration {}.'.format(iteration+1))
                    print('-'*100)
                elif self.verbose:
                    if info: print('[*] {}'.format(info))
                    print('[*] Go to repeated passes.')
            else:
                # Repeated pass iterates over non-boundary examples.
                I_non_boundary = np.where(np.logical_and(np.absolute(self.alpha) > 0, np.absolute(self.alpha) < self.C) == True)[0].tolist()
                if len(I_non_boundary):
                    E_list = np.vectorize(self._E)(I_non_boundary)
                    if not max(E_list) - min(E_list) < 1:
                        for i in I_non_boundary:
                            num_changed_alphas += self._examine_example(i)

                if self.verbose and num_changed_alphas == 0:
                    print('-'*100)
                    if info: print('[*] {}'.format(info))
                    print('[*] Repeated passes done. Go back to one pass.')

            if examine_all == 1:
                # One pass done, go to repeated passes.
                examine_all = 0
            elif num_changed_alphas == 0:
                # Repeated pass done, go back to one pass.
                examine_all = 1

            iteration += 1
            if self.verbose and (iteration == 1 or iteration % 100 == 0 or iteration == self.max_iteration):
                # Compute training and testing error
                train_error = self.scorer(y_truth=self.train_y, y_pred=self.hypothesis(X=self.train_X))
                test_error = self.scorer(y_truth=self.test_y, y_pred=self.hypothesis(X=self.test_X))
                print('-'*100)
                if info: print('[*] {}'.format(info))
                print('[*] {} alphas changed.'.format(num_changed_alphas))
                print('[*] Iteration: {} | Train error: {} | Test error: {}'.format(iteration, train_error, test_error))

            if iteration == self.max_iteration:
                print('-'*100)
                print('[*] Max iteration acheived.')
                break

        if self.verbose: print('[*] Averaging post-computed biases as final bias of SVR hypothesis.')
        self._postcompute_biases()

    def _violate_KKT_conditions(self, i):
        """ Check if an example violates the KKT conditons """

        alpha_i = self.alpha[i]
        E_i = self._E(i)

        # ---- Not the same as SVC ----
        if alpha_i == 0 and not (-self.epsilon <= E_i + self.tol and E_i <= self.epsilon + self.tol):
            return True
        if (-self.C < alpha_i and alpha_i < 0) and not E_i == self.epsilon:
            return True
        if (0 < alpha_i and alpha_i < self.C) and not E_i == -self.epsilon:
            return True
        if alpha_i == -self.C and not E_i >= self.epsilon - self.tol:
            return True
        if alpha_i == self.C and not E_i <= self.epsilon - self.tol:
            return True

        return False

    def _examine_example(self, i):
        """ Implement Platt's heuristics to select a good alpha pair to optimize.
            (First heuristic is not implemented since it makes training slower)
        """
        # Check if alpha_i needs updating (alpha_i violates KKT conditions)
        if self._violate_KKT_conditions(i):

            # Retrieve indexes of non boundary examples
            I_non_boundary = np.where(np.logical_and(np.absolute(self.alpha) > 0, np.absolute(self.alpha) < self.C) == True)[0].tolist()

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

        # ---- Not the same as SVC ----
        L = max(-self.C, alpha_i + alpha_j - self.C)
        H = min(self.C, alpha_i + alpha_j + self.C)

        # This will not make any progress.
        if L == H: return 0

        # Compute eta (second derivative of the Lagrange dual function = -eta)
        if self.enable_kernel_cache:
            eta = self.kernel_cache[i][i] + self.kernel_cache[j][j] - 2*self.kernel_cache[i][j]
        else:
            eta = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - 2*self.kernel(x_i, x_j)

        # eta > 0 => second derivative(-eta) < 0 => maximum exists.
        if eta <= 0: return 0

        # ---- Not the same as SVC ----

        # Although the update rule of `alpha_j` is a **function of itself**.
        # by analysis, we can still update `alpha_j` by trick, since there's only three possible `alpha_j_new`
        # See SMO supplement for more details.
        delta_E_ij = E_i - E_j

        # Calculate list of possible new alphas.
        # `x` is a function of `alpha_j_new` and it actually only takes one of {-2, 0, 2}
        possible_alpha_j_new = lambda x: alpha_j + (delta_E_ij + x*self.epsilon)/eta
        possible_alpha_j_new_pos2 = possible_alpha_j_new(2)
        possible_alpha_j_new_zero = possible_alpha_j_new(0)
        possible_alpha_j_new_neg2 = possible_alpha_j_new(-2)

        # How `alpha_j` is updated depends on various conditions of `r_ij = alpha_i + alpha_j`
        r_ij = alpha_i + alpha_j

        # Compute new alpha_j and clip it inside [L, H]. This is the update case when eta > 0
        if r_ij == 0:
            if possible_alpha_j_new_pos2 <= L:
                alpha_j_new = L
            elif L < possible_alpha_j_new_pos2 and possible_alpha_j_new_pos2 < 0:
                alpha_j_new = possible_alpha_j_new_pos2
            elif possible_alpha_j_new_neg2 >= H:
                alpha_j_new = H
            elif 0 < possible_alpha_j_new_neg2 and possible_alpha_j_new_neg2 < H:
                alpha_j_new = possible_alpha_j_new_neg2
            else:
                alpha_j_new = 0

        elif 0 < r_ij and r_ij < self.C:
            if possible_alpha_j_new_pos2 <= L:
                alpha_j_new = L
            elif L < possible_alpha_j_new_pos2 and possible_alpha_j_new_pos2 < 0:
                alpha_j_new = possible_alpha_j_new_pos2
            elif possible_alpha_j_new_zero <= 0:
                alpha_j_new = 0
            elif 0 < possible_alpha_j_new_zero and possible_alpha_j_new_zero < r_ij:
                alpha_j_new = possible_alpha_j_new_zero
            elif possible_alpha_j_new_neg2 >= H:
                alpha_j_new = H
            elif r_ij < possible_alpha_j_new_neg2 and possible_alpha_j_new_neg2 < H:
                alpha_j_new = possible_alpha_j_new_neg2
            else:
                alpha_j_new = r_ij

        elif r_ij == self.C:
            if possible_alpha_j_new_zero <= L:
                alpha_j_new = L
            elif L < possible_alpha_j_new_zero and possible_alpha_j_new_zero < H:
                alpha_j_new = possible_alpha_j_new_zero
            else:
                alpha_j_new = H

        elif r_ij > self.C:
            if possible_alpha_j_new_zero < L:
                alpha_j_new = L
            elif L <= possible_alpha_j_new_zero and possible_alpha_j_new_zero <= H:
                alpha_j_new = possible_alpha_j_new_zero
            else:
                alpha_j_new = H

        elif -self.C < r_ij and r_ij < 0:
            if possible_alpha_j_new_pos2 <= L:
                alpha_j_new = L
            elif L < possible_alpha_j_new_pos2 and possible_alpha_j_new_pos2 < r_ij:
                alpha_j_new = possible_alpha_j_new_pos2
            elif possible_alpha_j_new_zero <= r_ij:
                alpha_j_new = r_ij
            elif r_ij < possible_alpha_j_new_zero and possible_alpha_j_new_zero < 0:
                alpha_j_new = possible_alpha_j_new_zero
            elif possible_alpha_j_new_neg2 >= H:
                alpha_j_new = H
            elif 0 < possible_alpha_j_new_neg2 and possible_alpha_j_new_neg2 < H:
                alpha_j_new = possible_alpha_j_new_neg2
            else:
                alpha_j_new = 0

        elif r_ij == -self.C:
            if possible_alpha_j_new_zero <= L:
                alpha_j_new = L
            elif L < possible_alpha_j_new_zero and possible_alpha_j_new_zero < H:
                alpha_j_new = possible_alpha_j_new_zero
            else:
                alpha_j_new = H

        elif r_ij < -self.C:
            if possible_alpha_j_new_zero < L:
                alpha_j_new = L
            elif L <= possible_alpha_j_new_zero and possible_alpha_j_new_zero <= H:
                alpha_j_new = possible_alpha_j_new_zero
            else:
                alpha_j_new = H

        # Compute new alpha_i based on new alpha_j
        alpha_i_new = alpha_i - (alpha_j_new - alpha_j)

        # Compute step sizes
        delta_alpha_i = alpha_i_new - alpha_i
        delta_alpha_j = alpha_j_new - alpha_j

        # Update weight vector
        if self.use_w:
            self.w = self.w + delta_alpha_i*x_i + delta_alpha_j*x_j

        # Update b
        if self.enable_kernel_cache:
            b_i = self.b - E_i - delta_alpha_i*self.kernel_cache[i][i] - delta_alpha_j*self.kernel_cache[i][j]
            b_j = self.b - E_j - delta_alpha_i*self.kernel_cache[i][j] - delta_alpha_j*self.kernel_cache[j][j]
        else:
            b_i = self.b - E_i - delta_alpha_i*self.kernel(x_i, x_i) - delta_alpha_j*self.kernel(x_i, x_j)
            b_j = self.b - E_j - delta_alpha_i*self.kernel(x_i, x_j) - delta_alpha_j*self.kernel(x_j, x_j)
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
        """ Linear classifier `f(x) = wx + b`, used when training or making predictions.
            @param `X`: `X` can be a single example with shape(D,) or multiple examples with shape(N, D)
        """
        # Use b_mean to make predictions when training is done.
        b = self.b_mean if self.b_mean is not None else self.b

        if self.use_w:
            # Speed up by using computed weight only when linear kernel is used.
            return np.dot(X, self.w) + b
        else:
            # If X is single example
            if X.ndim == 1:
                # ---- Not the same as SVC ----
                return np.dot(self.alpha, self.kernel(self.train_X, X)) + b
            # Multiple examples
            elif X.ndim == 2:
                return np.array([np.dot(self.alpha, self.kernel(self.train_X, _X)) + b for _X in X])

    def _E(self, i):
        """ Prediction error: _f(x_i) - y_i, used when training. """
        if self.enable_kernel_cache:
            # ---- Not the same as SVC ----
            return np.dot(self.alpha, self.kernel_cache[i]) + self.b - self.train_y[i]
        else:
            return self._f(self.train_X[i]) - self.train_y[i]

    def _precompute_kernel_cache(self):
        """ If self.enable_kernel_cache is True, then precompute kernel results for all training examples.
            This can speed up training but need time to initialize when data is large.
        """
        kernel_cache = np.zeros((len(self.train_X), len(self.train_X)))
        for i, x_i in enumerate(self.train_X):
            for j, x_j in enumerate(self.train_X):
                kernel_cache[i][j] = self.kernel(x_i, x_j)
        return kernel_cache

    def _postcompute_biases(self):
        """ Post-computed biases for non-boundary training examples (support vectors) when training is done.
            This is for estimating sample mean and sample std of biases.
            For a good learning result, sample std of biases should be small.
        """
        # ---- Not the same as SVC ----
        def _b(i):
            if self.enable_kernel_cache:
                return self.train_y[i] - np.dot(self.alpha, self.kernel_cache[i])
            else:
                return self.train_y[i] - self._f(self.train_X[i])

        I_non_boundary = np.where(np.logical_and(np.absolute(self.alpha) > 0, np.absolute(self.alpha) < self.C) == True)[0].tolist()

        if len(I_non_boundary):
            biases = np.vectorize(_b)(I_non_boundary)
            self.b_mean = np.mean(biases)
            self.b_std = np.sqrt(np.sum((biases - self.b_mean)**2) / (len(biases) - 1))
            self.postcomputed_biases[I_non_boundary] = biases

    def _linear_kernel(self, X, x):
        """ Linear kernel:
            @param `X`: `X` can be a single example with shape(D,) or multiple examples with shape(N, D)
            @param `x`: `x` can only be a single example with shape(D,)
        """
        return np.dot(X, x)

    def _poly_kernel(self, X, x):
        """ Polynomial kernel:
            @param `X`: `X` can be a single example with shape(D,) or multiple examples with shape(N, D)
            @param `x`: `x` can only be a single example with shape(D,)
        """
        return (1 + np.dot(X, x))**self.poly_degree

    def _rbf_kernel(self, X, x):
        """ RBF (guassian) kernel:
            @param `X`: `X` can be a single example with shape(D,) or multiple examples with shape(N, D)
            @param `x`: `x` can only be a single example with shape(D,)
        """
        # If X is single example
        if X.ndim == 1:
            sqrt_norm = np.linalg.norm(X - x)**2
        # Multiple examples
        elif X.ndim == 2:
            sqrt_norm = np.linalg.norm(X - x, axis=1)**2

        return np.exp(-sqrt_norm / (2.0 * (self.rbf_sigma**2)))
