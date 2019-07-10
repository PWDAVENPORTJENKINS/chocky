"""
P. W. Davenport-Jenkins
University of Manchester
MSc Econometrics
2019-07-08
"""
import warnings
# import multiprocessing as mp
# from numba import jit
import scipy.optimize as optimize
from functools import partial
# import scipy.stats as stats
import numpy as np
# from autograd import grad, jacobian
import math
import time
# import seaborn as sns
from scipy.optimize import LinearConstraint
import operator
warnings.filterwarnings("error")


"""
This makes the grid for which we perform the grid search
"""
def make_grid(anchor, mesh_size, size):

    upper = [anchor + mesh_size * i for i in range(size)]
    lower = [anchor - mesh_size * (1+i) for i in range(size)]
    array = list(reversed(lower)) + upper

    return array


def moment_conditions(beta, x, y, z):
    return (np.add(y, -beta * x) * z).T


def GMM_weighting_matrix(moment_conditions):

    K = len(moment_conditions[0])

    T = len(moment_conditions)

    omega = np.zeros((K, K))

    for moment_vector in moment_conditions:

        omega = omega + np.outer(moment_vector, moment_vector)

    return np.linalg.inv(omega/T)


def GMM_objective_function(beta, x, y, z, weighting_matrix):

    average = moment_conditions(beta, x, y, z).mean(axis=0)

    return average.T @ weighting_matrix @ average


def GMM(beta_initial, x, y, z, K):

    # First Stage: calculate initial beta
    beta_1 = optimize.minimize(
            GMM_objective_function,
            beta_initial,
            args=(x, y, z, np.identity(K)),
            method="BFGS"
    )

    beta_1 = beta_1.x

    # Use this value to compute the optimal weighting matrix
    weighting_matrix = GMM_weighting_matrix(
                                moment_conditions(beta_1, x, y, z)
    )

    # Second stage:: use the optimal weighting matrix to compute 2S-GMM
    # estimator of beta
    beta_2 = optimize.minimize(
            GMM_objective_function,
            beta_1,
            args=(x, y, z, weighting_matrix),
            method="BFGS"
    )

    return beta_2.x


def EL(value):

    return_value = (
        np.log(2) + 2 * (value - 0.5) + 2 * (value - 0.5)**2 +
        2.666 * (value - 0.5)**3 + 4 * (value - 0.5)**4
    )

    # When the value is less than 0.5, use the analytic formula
    # ln(1 - value).
    try:
        return_value[value < 0.5] = np.log(1 - value[value < 0.5])
    except RuntimeWarning:
        return_value = (
            np.log(2) + 2 * (value - 0.5) + 2 * (value - 0.5)**2 +
            2.666 * (value - 0.5)**3 + 4 * (value - 0.5)**4
        )

    return return_value


"""
I apply a function to an array.
Some values in that array may be equal to 1
however that is probably unlikely.
What is likely however is that the values are near 1
and so the jacobian will explode in size.

Should I use  Taylor series thing like above?
"""
def EL_derivative(value):
    try:
        return 1/(1-value)
    except ZeroDivisionError:
        return 0 * value


def ET(value):

    return -np.exp(value)


def ET_cost_function(moments, l):

    return -np.mean(ET(moments @ l))


def EL_cost_function(moments, l):

    return -np.mean(EL(moments @ l))


def ET_jacobian(moments, l):

    q = ET(moments @ l)
    j = moments * q[:, np.newaxis]
    return -(np.ones(len(q)) @ j) / len(q)


"""
Doesn't work with SLQLP method and with other methods doesn't seem to
help speed it up. Could be completly wrong formula in fairness...
"""
def ET_hessian(moments, l):

    outer = np.zeros((len(l), len(l)))

    q = ET(moments @ l)

    j = moments * q[:, np.newaxis]

    for i in range(len(q)):
        outer = outer + np.outer(j[i], moments[i])

    return -outer / len(q)


def EL_jacobian(moments, l):

    q = EL_derivative(moments @ l)
    j = moments * q[:, np.newaxis]
    return -(np.ones(len(q)) @ j) / len(q)


def GEL_ET(beta, x, y, z, K):

    """
    Here moments_matrix is a T x K matrix
    """
    moments_matrix = moment_conditions(beta, x, y, z)

    initial_params = np.zeros(K)

    cost = partial(ET_cost_function, moments_matrix)

    jac_ET = partial(ET_jacobian, moments_matrix)

    result = optimize.minimize(
        cost,
        initial_params,
        method="SLSQP",
        jac=jac_ET,
        options={'maxiter': 100}
    )
    return result.x


def GEL_EL(beta, x, y, z, K):

    """
    Here moments_matrix is a T x K matrix
    """
    moments_matrix = moment_conditions(beta, x, y, z)

    initial_params = np.zeros(K)

    cost = partial(EL_cost_function, moments_matrix)

    epsilon = 1e-6

    assert 0 < epsilon, 'Function goes to inf as eps goes to 0.'
    constraint = LinearConstraint(
        moments_matrix,
        ub=np.ones(moments_matrix.shape[0]) - epsilon,
        lb=-np.inf
    )

    """
    Including the Jacobian for the EL version seems to slow my code down
    dramatically.
    """
    # jac_EL = partial(EL_jacobian, moments_matrix)
    result = optimize.minimize(
        cost,
        initial_params,
        constraints=constraint,
        # jac=jac_EL,
        method='SLSQP',
        options={'maxiter': 100}
    )
    return result.x


def grid_search(beta_grid, x, y, z, K, method):
    grid_size = len(beta_grid)
    lambda_dictionary = dict()
    for i in range(grid_size):
            beta = beta_grid[i]
            if method == "ET":
                lambda_beta = GEL_ET(beta, x, y, z, K)
            if method == "EL":
                lambda_beta = GEL_EL(beta, x, y, z, K)
            lambda_dictionary[i] = lambda_beta

    objective_dict = dict()
    for i in range(grid_size):
            beta = beta_grid[i]
            moments = moment_conditions(beta, x, y, z)
            lambda_beta = lambda_dictionary[i]
            if method == "ET":
                value = -ET_cost_function(moments, lambda_beta)
            if method == "EL":
                value = -EL_cost_function(moments, lambda_beta)
            objective_dict[i] = value

    return objective_dict, lambda_dictionary, beta_grid


def monte_carlo_simulation(T,
                           K,
                           R_SQUARED,
                           RHO,
                           BETA,
                           GMM_list,
                           ET_list,
                           ET_lambda_list,
                           EL_list,
                           EL_lambda_list):

    beta_guess = 0.5
    eta = math.sqrt(R_SQUARED / (K * (1 - R_SQUARED)))
    pi = eta * np.ones(K)
    error_mean = np.array([0, 0])
    error_varcov = np.array([[1, RHO], [RHO, 1]])

    z_mean = np.zeros(K)
    z_varcov = np.identity(K)
    z = np.random.multivariate_normal(z_mean, z_varcov, T).T
    epsilon, nu = np.random.multivariate_normal(error_mean, error_varcov, T).T

    x = z.T @ pi + nu
    y = BETA * x + epsilon

    beta_GMM = float(GMM(beta_guess, x, y, z, K))

    beta_grid = make_grid(BETA, 0.01, 100)
    #
    dict_to_max_ET, lambda_dictionary_ET, beta_ETs = grid_search(
                                            beta_grid,
                                            x,
                                            y,
                                            z,
                                            K,
                                            "ET"
    )

    dict_to_max_EL, lambda_dictionary_EL, beta_ELs = grid_search(
                                            beta_grid,
                                            x,
                                            y,
                                            z,
                                            K,
                                            "EL"
    )

    index_ET = min(dict_to_max_ET.items(), key=operator.itemgetter(1))[0]
    lambda_ET = lambda_dictionary_ET[index_ET]
    beta_ET = beta_ETs[index_ET]

    index_EL = min(dict_to_max_EL.items(), key=operator.itemgetter(1))[0]
    lambda_EL = lambda_dictionary_EL[index_EL]
    beta_EL = beta_ELs[index_EL]

    GMM_list.append(beta_GMM)
    EL_list.append(beta_EL)
    EL_lambda_list.append(lambda_EL)
    ET_list.append(beta_ET)
    ET_lambda_list.append(lambda_ET)

    return [GMM_list, ET_list, ET_lambda_list, EL_list, EL_lambda_list]


"""
Monte Carlo Simulation
Values I want to run simulations for.

Sample Size: T
    T: 100, 200

Number of Instruments: K
    K: 1, 5, 10, 20, 50

R Squared Value: R_SQUARED
    R_SQUARED: 0.001, 0.01, 0.1

Correlation of errors: RHO
    RHO: 0, 0.25, 0.50, 0.75, 0.95

Replications: N
    N: 5000

"""

GMM_list = list()  # the GMM estimates
ET_list = list()  # the ET estimates
EL_list = list()  # the EL estimates
ET_lambda_list = list()  # the lagrange multipliers from ET
EL_lambda_list = list()  # the lagrange multipliers from EL

T = 100
K = 5
R_SQUARED = 0.01
RHO = 0.25

BETA = 0
start_time = time.time()
for i in range(5):  # want this to be N=5000
    monte_carlo_simulation(T,
                           K,
                           R_SQUARED,
                           RHO,
                           BETA,
                           GMM_list,
                           ET_list,
                           ET_lambda_list,
                           EL_list,
                           EL_lambda_list)
    print("DONE %s-th replication" % (i))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

print("--- Total Time: %s seconds ---" % (time.time() - start_time))
