import sys
import time
import math
import random
import argparse
import numpy as np

from PAC import *

parser = argparse.ArgumentParser(description='Learning an axis-aligned rectangle from the given 2-D points sampled from normal distribution.')
parser.add_argument('--delta', dest='delta', type=float, default=0.01, help='Probability of generalization error upper bounded by <eps> is at least confidence = 1 - delta (default: 0.01)')
parser.add_argument('--eps', dest='eps', type=float, default=0.1, help='Upper bound of generalization error (default: 0.1)')
parser.add_argument('--mean_x', dest='mean_x', type=float, default=0.0, help='Mean of x-coordinate for bivariate normal distribution (default: 0.0)')
parser.add_argument('--mean_y', dest='mean_y', type=float, default=0.0, help='Mean of y-coordinate for bivariate normal distribution (default: 0.0)')
parser.add_argument('--std_x', dest='std_x', type=float, default=1.0, help='Standard deviation of x-coordinate for bivariate normal distribution (default: 1.0)')
parser.add_argument('--std_y', dest='std_y', type=float, default=1.0, help='Standard deviation of y-coordinate for bivariate normal distribution (default: 1.0)')
parser.add_argument('--r_xy', dest='r_xy', type=float, default=0.5, help='Correlation coefficient of x-coordinate and y-coordinate for bivariate normal distribution (default: 0.5)')
parser.add_argument('--rect_min_x', dest='rect_min_x', type=float, default=None, help='Bottom left point x of unknown concept axis-aligned rectangle (default: random)')
parser.add_argument('--rect_min_y', dest='rect_min_y', type=float, default=None, help='Bottom left point y of unknown concept axis-aligned rectangle (default: random)')
parser.add_argument('--rect_max_x', dest='rect_max_x', type=float, default=None, help='Top right point x of unknown concept axis-aligned rectangle (default: random)')
parser.add_argument('--rect_max_y', dest='rect_max_y', type=float, default=None, help='Top right point y of unknown concept axis-aligned rectangle (default: random)')
parser.add_argument('--verbose', dest='verbose', type=bool, default=False, help='Whether print the detailed process of verifying generalization guarantee (default: False)')

args = parser.parse_args()

if __name__ == '__main__':

    # Assign arguments
    delta = args.delta
    eps = args.eps
    mean_x = args.mean_x
    mean_y = args.mean_y
    std_x = args.std_x
    std_y = args.std_y
    r_xy = args.r_xy
    rect_min_x = args.rect_min_x
    rect_min_y = args.rect_min_y
    rect_max_x = args.rect_max_x
    rect_max_y = args.rect_max_y
    verbose = args.verbose

    mean = [mean_x, mean_y]
    cov = [[std_x**2, r_xy*std_x*std_y], [r_xy*std_x*std_y, std_y**2]]

    # Print arguments information
    print('-'*100)
    print('[*] Input arguments:')
    print('    - delta: {}'.format(delta))
    print('    - epsilon: {}'.format(eps))
    print('    - mean_x: {}, mean_y: {}'.format(mean_x, mean_y))
    print('    - std_x: {}, std_y: {}, r_xy: {}'.format(std_x, std_y, r_xy))
    print('    - corvariance matrix: {}'.format(cov))
    print('    - verbose (whether print the detailed process of verifying generalization guarantee): {}'.format(verbose))
    print('-'*100)

    # Handle unknown target concept
    N = int(math.ceil((1.8595/eps)**2))
    X_N = np.random.multivariate_normal(mean=mean, cov=cov, size=N)

    if rect_min_x == None or rect_min_y == None or rect_max_x == None or rect_max_y == None:
        print('[*] Generating concept from sample size N_eps = {}...'.format(N))
        # Generate unknown target concept (axis-aligned rectangle) from sample size N
        R_target = generate_target_rectangle(X_N, eps)
    else:
        R_target = Rectangle(min_x=rect_min_x,
                             min_y=rect_min_y,
                             max_x=rect_max_x,
                             max_y=rect_max_y)
        # Estimate probability of unknown target concept with empirical probability
        # Size of data from normal distribution must be ceil((1.8595/eps)**2)
        p_hat = estimate_prob_of_target_rectangle(X_N, R_target)

        if p_hat >= 3*eps:
            print('[*] Requirement P(c) >= {} is qualified. p_hat ({}) >= {}'.format(2*eps, p_hat, round(3*eps, 2)))
        else:
            print('[!] Requirement P(c) >= {} is not qualified. p_hat ({}) < {}'.format(2*eps, p_hat, round(3*eps, 2)))
            print('[!] Exit program.')
            sys.exit(0)
    print('[*] Concept: {}'.format(R_target))
    print('-'*100)

    # Generate our good hypothesis to approximate unknown target concept from sample size m
    # m = int(math.ceil((4/eps) * math.log(4/delta)))
    m = 50
    X_m = np.random.multivariate_normal(mean=mean, cov=cov, size=m)
    print('[*] Generating hypothesis from sample size m = {}...'.format(m))
    R_hypothesis = generate_hypothesis_rectangle(X_m, R_target)
    print('[*] Hypothesis: {}'.format(R_hypothesis))

    # Estimate generalization error of hypothesis
    M = int(math.ceil((19.453/eps)**2))
    X_M = np.random.multivariate_normal(mean=mean, cov=cov, size=M)
    print('[*] Computing generalization error from sample size M_eps = {}...'.format(M))
    generalization_error = estimate_generalization_error(X_M, R_target, R_hypothesis)
    print('[*] Generalization error ({})'.format(generalization_error))
    print('-'*100)

    # Verify generalization guarantee
    print('[*] Verifying generalization guarantee...')

    times = int(math.ceil(10 / delta))
    error_time = 0

    for t in range(1, times+1):
        start_time = time.time()

        # Generate our good hypothesis to approximate unknown target concept from sample size m
        m = int(math.ceil((4/eps) * math.log(4/delta)))
        X_m = np.random.multivariate_normal(mean=mean, cov=cov, size=m)
        R_hypothesis = generate_hypothesis_rectangle(X_m, R_target)

        # Estimate generalization error of hypothesis
        M = int(math.ceil((19.453/eps)**2))
        X_M = np.random.multivariate_normal(mean=mean, cov=cov, size=M)
        generalization_error = estimate_generalization_error(X_M, R_target, R_hypothesis)

        if verbose:
            print('[*] Progress: {}/{}'.format(t, times))
            print('[*] Execution time: {} seconds'.format(time.time() - start_time))

            if generalization_error > eps:
                print('[!] Generalization error ({}) > eps ({})'.format(generalization_error, eps))
                error_time += 1
            else:
                print('[*] Generalization error ({}) <= eps ({})'.format(generalization_error, eps))
            print('-'*100)

    if error_time <= 10:
        print('[*] The generalization guarantee is valid. Error time: {}/{} <= 10/{}'.format(error_time, times, times))
    else:
        print('[!] The generalization guarantee does not hold. Error time: {}/{} > 10/{}'.format(error_time, times, times))
    print('-'*100)
