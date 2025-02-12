import pytest
import numpy as np
import cimr_rgb.iterative_methods as itmet


variable1 = dict()
samples1  = dict()
exp1      = dict()
variable1['varA'] = np.array([0.1, 0.7, 1. , 0.2, 0.5])
variable1['varB'] = np.array([1.3, 8.4, 9.7, 9.3, 6.1])
variable1['varC'] = np.array([4.9, 0.9, 4.5, 2.4, 3.1])
samples1['indexes']   = np.array([4, 3, 0, 0, 3])
exp1['varA'] = np.array([0.5, 0.2, 0.1, 0.1, 0.2])                                                           


A1 = np.array([[4., 1.], [1., 3.]])
Y1 = np.array([1, 2])
X1 = np.linalg.solve(A1, Y1)

A2 = np.array([[1., 0.99], [0.99, 0.98]])
Y2 = np.array([1., 1.])

A_random = np.array([[0.36, 0.3 , 0.43, 0.31, 0.35],
                     [0.4 , 0.04, 0.06, 0.63, 0.58],
                     [0.78, 0.86, 0.07, 0.25, 0.25],
                     [0.02, 0.47, 0.57, 0.78, 0.34],
                     [0.31, 0.43, 0.68, 0.9 , 0.21]])
A3 = A_random @ A_random.T  # Ensure positive-definiteness
Y3 = np.array([0.64, 0.73, 0.39, 0.97, 0.43])

@pytest.mark.parametrize(
    ("A", "Y", "lambda_param", "alpha", "n_iter", "rtol"),
    [
        pytest.param(A1, Y1, 0., None, 10000, 1e-3, id='exact_solution'),
        pytest.param(A1, Y1, 0.1, None, 10000, 1e-3, id='exact_solution_big_regularization'),
        pytest.param(A2, Y2, 0., None, 10000, 1e-3, id='ill_conditioned_system'),
        pytest.param(A3, Y3, 0., 0.1, 10000, 1e-3, id='large_positive_definite')
    ]
)
def test_landweber(A, Y, lambda_param, alpha, n_iter, rtol):

    X, count = itmet.landweber(A, Y, lambda_param, alpha, n_iter, rtol)
    ares = np.linalg.norm(A@X-Y)
    rres = ares//np.linalg.norm(Y)
    if count < n_iter:
        if ares < lambda_param:
            assert ares<=lambda_param, f"Iteration exited with atol {ares}, while should be less than {lambda_param}."
        else:
            assert rres<=rtol, f"Iteration exited with rtol {rres}, while should be less than {rtol}."
    else:
        assert True #cannot say anything if convergence is not reached


@pytest.mark.parametrize(
    ("A", "Y", "lambda_param", "n_iter", "rtol"),
    [
        pytest.param(A1, Y1, 0., 10000, 1e-3, id='exact_solution'),
        pytest.param(A1, Y1, 0.1, 10000, 1e-3, id='exact_solution_big_regularization'),
        pytest.param(A2, Y2, 0., 10000, 1e-3, id='ill_conditioned_system'),
        pytest.param(A3, Y3, 0., 10000, 1e-3, id='large_positive_definite')
    ]
)
def test_cgne(A, Y, lambda_param, n_iter, rtol):

    X, count = itmet.conjugate_gradient_ne(A, Y, lambda_param, n_iter, rtol)
    ares = np.linalg.norm(A@X-Y)
    rres = ares//np.linalg.norm(Y)
    if count < n_iter:
        if ares < lambda_param:
            assert ares<=lambda_param, f"Iteration exited with atol {ares}, while should be less than {lambda_param}."
        else:
            assert rres<=rtol, f"Iteration exited with rtol {rres}, while should be less than {rtol}."
    else:
        assert True #cannot say anything if convergence is not reached