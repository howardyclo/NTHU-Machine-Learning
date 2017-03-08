""" PAC learning framework """

import math
import random
import numpy as np

class Rectangle:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.width = max_x - min_x
        self.height = max_y - min_y

    def in_rectangle(self, X):
        return np.array((X[:,0] >= self.min_x) & (X[:,0] <= self.max_x) & (X[:,1] >= self.min_y) & (X[:,1] <= self.max_y))

    def __str__(self):
        return 'bottom left point ({},{}) ; top right point ({},{})'.format(self.min_x, self.min_y, self.max_x, self.max_y)

def estimate_prob_of_target_rectangle(X, R_target):
    # Classify points
    labels = R_target.in_rectangle(X)

    # Estimate probability of unknown target concept with empirical probability
    # Size of data from normal distribution must be ceil((1.8595/eps)**2)
    p_hat = np.mean(labels)

    return p_hat

def generate_target_rectangle(X, eps):
    """ Generate unknown target concept that p_hat(c) >= 3*eps """

    while True:
        # Generate corner points of rectangle
        min_x = np.min(X[:,0])
        min_y = np.min(X[:,1])
        mid_x = np.median(X[:,0])
        mid_y = np.median(X[:,1])
        max_x = np.max(X[:,0])
        max_y = np.max(X[:,1])

        rect_min_x = random.randrange(int(min_x), int(mid_x))
        rect_min_y = random.randrange(int(min_y), int(mid_y))
        rect_max_x = random.randrange(int(mid_x), int(max_x))
        rect_max_y = random.randrange(int(mid_y), int(max_y))

        # Generate unknown target concept
        R_target = Rectangle(min_x=rect_min_x,
                             min_y=rect_min_y,
                             max_x=rect_max_x,
                             max_y=rect_max_y)

        # Estimate probability of unknown target concept with empirical probability
        # Size of data from normal distribution must be ceil((1.8595/eps)**2)
        p_hat = estimate_prob_of_target_rectangle(X, R_target)

        if p_hat >= 3*eps:
            print('[*] Requirement P(c) >= {} is qualified. p_hat ({}) >= {}'.format(2*eps, p_hat, round(3*eps, 2)))
            print('[*] Successfully generate concept.')
            return R_target
        else:
            print('[!] Requirement P(c) >= {} is not qualified. p_hat ({}) < {}'.format(2*eps, p_hat, round(3*eps, 2)))
            print('[!] Re-generate concept...')


def generate_hypothesis_rectangle(X, R_target):
    """ Algorithm for generating a good hypothesis from hypothesis set.
        In our case, the hypothesis set is same as the concept class,
        that is, all the possible rectangles in 2-D plane.
        Our goal is to learn a good rectangle that is approximate to our target unknown rectangle.
        Thus, our strategy is to select the "tightest" rectangle containing the points in our target unknown rectangle.
    """

    # Classify points
    labels = R_target.in_rectangle(X)

    # Select the tightest corner points in unknown target rectangle
    tight_rect_min_x = np.min(X[labels == 1, 0])
    tight_rect_min_y = np.min(X[labels == 1, 1])
    tight_rect_max_x = np.max(X[labels == 1, 0])
    tight_rect_max_y = np.max(X[labels == 1, 1])

    # Generate our good hypothesis
    R_hypothesis = Rectangle(min_x=tight_rect_min_x,
                             min_y=tight_rect_min_y,
                             max_x=tight_rect_max_x,
                             max_y=tight_rect_max_y)

    return R_hypothesis

def estimate_generalization_error(X, R_target, R_hypothesis):
    # Generate test true labels
    label_true = R_target.in_rectangle(X)

    # Make predictions
    label_pred = R_hypothesis.in_rectangle(X)

    # Misclassified labels
    label_error = (label_true != label_pred)

    # Calculate error
    generalization_error = np.mean(label_error)

    return generalization_error
