import itertools
import random
import time

import seaborn

from benchmarks.utils import benchmark_parameterized, find_polynomial_coefficients
from tests.helpers.utils import generate_message

def randint_generator(a, b):
    while True:
        yield random.randint(a, b)

def test_function_to_benchmark(n_term, n2_term):
    O = itertools.product(
        range(n2_term),
        range(n2_term),
        range(n_term),
    )
    for _ in O:
        time.sleep(0.0001)

params = {
    "n_term" : randint_generator(0, 10),
    "n2_term" : randint_generator(0, 10),
}

# Run benchmark
results = benchmark_parameterized(
    fn = test_function_to_benchmark,
    parameters = params,
    iterations = 1_000,
    repitions = 5,
)

# Analyze data
poly_coef = find_polynomial_coefficients(results)

# Create heat map and output
seaborn.set_theme()
plot = seaborn.heatmap(poly_coef, annot=True)
plot.figure.savefig('report.png')
