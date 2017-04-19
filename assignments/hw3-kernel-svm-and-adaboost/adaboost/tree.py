import numpy as np

from utils import zero_one_loss

class ShallowDecisionTree:
    def __init__(self, data, sample_weights=None, scorer=zero_one_loss, verbose=False):
        self.X = data['X']
        self.y = data['y']
        self.sample_weights = sample_weights if sample_weights is not None else np.ones_like(self.y).astype(float) / len(self.y)
        self.param = {'feature_index': None, 'split_point': None, 'label': None}
        self.scorer = scorer
        self.verbose = verbose

    def train(self):

        best_error = np.inf
        binary_class_label = list(set(self.y))

        # Iterate over all features
        for feature_index in range(self.X.shape[1]):
            if self.verbose:
                print('-'*100)
                print('[*] Scanning Feature: {} ...'.format(feature_index))

            # Find best split (hypothesis) on data w.r.t the current feature
            x = self.X[:, feature_index]
            split_points = list(set(x))

            for split_point in split_points:
                # Choost the hypothesis either classifies one side +1, the other side -1
                # or classifies one side -1, the other size +1, which leads to lowest weighted error.
                hypothesis1 = np.vectorize(lambda val: binary_class_label[0] if val >= split_point else binary_class_label[1])
                hypothesis2 = np.vectorize(lambda val: binary_class_label[1] if val >= split_point else binary_class_label[0])

                y_pred1 = hypothesis1(x)
                y_pred2 = hypothesis2(x)

                error1 = self.scorer(y_truth=self.y, y_pred=y_pred1, sample_weights=self.sample_weights)
                error2 = self.scorer(y_truth=self.y, y_pred=y_pred2, sample_weights=self.sample_weights)

                if error1 < error2 and error1 < best_error:
                    best_error = error1
                    self.param.update({'feature_index': feature_index,
                                       'split_point': split_point,
                                       'label': binary_class_label[0]})
                    if self.verbose:
                        print('-'*100)
                        print('[*] Error: {}'.format(best_error))
                        print('[*] Update hypothesis: {}'.format(self.param))

                elif error2 < error1 and error2 < best_error:
                    best_error = error2
                    self.param.update({'feature_index': feature_index,
                                       'split_point': split_point,
                                       'label': binary_class_label[1]})
                    if self.verbose:
                        print('-'*100)
                        print('[*] Error: {}'.format(best_error))
                        print('[*] Update hypothesis: {}'.format(self.param))

        if self.verbose:
            print('-'*100)
            print('[*] Best Error: {}'.format(best_error))
            print('[*] Best hypothesis: {}'.format(self.param))
            print('{}'.format(self))
            print('-'*100)

    def hypothesis(self, X):
        x = X[:, self.param['feature_index']]
        the_other_label = list(set(self.y) - set([self.param['label']]))[0]
        return np.vectorize(lambda val: self.param['label'] if val >= self.param['split_point'] else the_other_label)(x)

    def __str__(self):
        the_other_label = list(set(self.y) - set([self.param['label']]))[0]

        description = \
        ' -  Label {} if value at {}-th feature is >= {}. \n' + \
        ' -  Label {} if value at {}-th feature is < {}.'

        return description.format(self.param['label'],
                                  self.param['feature_index'],
                                  self.param['split_point'],
                                  the_other_label,
                                  self.param['feature_index'],
                                  self.param['split_point'])
