import itertools
import multiprocessing
import time

import tqdm
from sklearn import linear_model
import pandas
import numpy

def time_function(fn, parameters, repititions):
    total_time = 0

    for _ in range(repititions):
        start = time.time()
        fn(**parameters)
        end = time.time()
        total_time += end - start

    return {
        "time" : total_time / repititions,
        **parameters,
    }

def benchmark_parameterized(
        fn,
        parameters,
        iterations = 100,
        repititions = 1,
):
    """
    Run benchmark on a specified function with specified parameters.

    Parameters is a dict of function kwargs with a random generator for possible
    values. Example: {"num_edges" : [5,10,15], "num_nodes" : [1,2,4]}

    Benchmarking every combination of parameters is likely to take
    a long time, so what we can do is sample from the given parameters
    N times where N = iterations.

    In addition, there might be variance between test runs so this
    function supports specifying the number of times to repeat.
    """

    # Create generator that generates all parameter combinations
    parameter_generator = (
        dict(zip(parameters.keys(), values))
        for values in zip(*parameters.values())
    )
    # Limit generator to number of iterations
    parameter_generator = itertools.islice(
        parameter_generator,
        iterations
    )

    # Generator that includes parameters and function under test
    time_function_generator = (
        (fn, p, repititions)
        for p in parameter_generator
    )

    pool = multiprocessing.Pool()

    # tqdm shows a progress bar
    return pool.starmap(
        time_function,
        tqdm.tqdm(
            time_function_generator,
            total = iterations,
            desc = "Running Benchmark"
        ),
        chunksize = 2
    )


def find_polynomial_coefficients(results):
    """
    Given benchmark data fit a multivaraite polynomial regression
    so that we can determine the complexity class
    """

    param_names = set(results[0].keys())
    param_names.remove("time")

    results_df = pandas.DataFrame(results)

    X = results_df.filter(param_names).to_numpy()
    Y = results_df["time"].to_numpy()

    # Generate polynomial features
    # Assume nothing greater than O(n^3)
    poly_degrees = range(1,4)
    X_ = numpy.hstack((X**(i) for i in poly_degrees))

    # Generate the regression object
    clf = linear_model.LinearRegression()

    # Perform the regression
    clf.fit(X_, Y)

    # Load coefficients so we can display them
    coef_df = pandas.DataFrame(clf.coef_.reshape((len(poly_degrees), len(param_names))))

    # Label axes
    coef_df = coef_df.set_axis(
        param_names,
        axis = 'columns',
    )
    coef_df = coef_df.set_axis(
        (f"x^{n}" for n in poly_degrees),
        axis = 'index',
    )

    return coef_df

