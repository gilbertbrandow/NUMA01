import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Callable
from interval import Interval, VectorizedInterval, BaseInterval
import time


def task_4(I1: BaseInterval, I2: BaseInterval) -> None:
    """
    Perform basic arithmetic operations (addition, subtraction, multiplication, division) on two intervals
    and print the results.

    :param I1: First interval.
    :param I2: Second interval.
    :return: None
    """
    sum: BaseInterval = I1 + I2
    diff: BaseInterval = I1 - I2
    product: BaseInterval = I1 * I2
    quotient: BaseInterval = I1 / I2

    print(sum)
    print(diff)
    print(product)
    print(quotient)


def task_10(
    p: Callable[[BaseInterval], BaseInterval],
    x1: npt.NDArray[np.floating],
    dx: float
) -> None:
    """
    Evaluate interval bounds for a given function over a range and plot the results.

    :param p: Function that operates on intervals.
    :param x1: Array of initial x-values (lower bounds for intervals).
    :param dx: Interval width (added to x1 to create upper bounds).
    :return: None
    """
    y1, yu = evaluate_interval_bounds(p=p, x1=x1, dx=dx)

    plt.figure("Task 10")
    plt.plot(x1, y1, label='y1')
    plt.plot(x1, yu, label='yu')
    plt.legend()
    plt.show()


def evaluate_interval_bounds(
    p: Callable[[BaseInterval], BaseInterval],
    x1: npt.NDArray[np.floating],
    dx: float
) -> tuple[list[float], list[float]]:
    """
    Evaluate a function on intervals defined by x1 and x1 + dx and return lower and upper bounds.

    :param p: Function that accepts an interval and returns an interval.
    :param x1: Array of initial x-values (lower bounds for intervals).
    :param dx: Interval width (added to x1 to create upper bounds).
    :return: Tuple containing two lists - (lower bounds, upper bounds).
    """
    xu: npt.NDArray[np.floating] = x1 + dx
    iv: list[BaseInterval] = [p(Interval(l, u)) for l, u in zip(x1, xu)]

    y1: list[float] = [I.a for I in iv]
    yu: list[float] = [I.b for I in iv]

    return (y1, yu)


def measure_time(f: Callable) -> float:
    """
    Measure and return the execution time of a given function.

    :param f: Function to be executed and timed.
    :return: Execution time of the function in seconds.
    """
    t_start: float = time.time()
    f()
    t_end: float = time.time()
    return t_end - t_start


def task_11(
    p: Callable[[BaseInterval], BaseInterval],
    x1: npt.NDArray[np.floating],
    dx: float
) -> None:
    """
    Evaluate vectorized intervals for a given function and plot the results.

    :param p: Function that operates on vectorized intervals.
    :param x1: Array of initial x-values (lower bounds for intervals).
    :param dx: Interval width (added to x1 to create upper bounds).
    :return: None
    """
    vi = evaluate_vectorized_intervals(p=p, x1=x1, dx=dx)

    plt.figure("Task 11")
    plt.plot(x1, vi.a, label='y1')
    plt.plot(x1, vi.b, label='yu')
    plt.legend()
    plt.show()


def evaluate_vectorized_intervals(
    p: Callable[[BaseInterval], BaseInterval],
    x1: npt.NDArray[np.floating],
    dx: float
) -> BaseInterval:
    """
    Evaluate a function on vectorized intervals defined by x1 and x1 + dx.

    :param p: Function that accepts a vectorized interval and returns a vectorized interval.
    :param x1: Array of initial x-values (lower bounds for intervals).
    :param dx: Interval width (added to x1 to create upper bounds).
    :return: Resulting vectorized interval containing lower and upper bounds.
    """
    xu: npt.NDArray[np.floating] = x1 + dx
    vi: BaseInterval = p(VectorizedInterval(x1, xu))
    return vi


def main() -> None:
    """
    Main function to execute tasks 4, 10, and 11, and measure execution time of interval evaluations.

    :return: None
    """
    task_4(I1=Interval(1, 4), I2=Interval(-2, -1))

    p: Callable[[BaseInterval], BaseInterval] = lambda x: 3 * \
        x**3 - 2*x**2 - 5*x - 1

    task_10(
        p=p,
        x1=np.linspace(0., 1, 1000),
        dx=0.5
    )
    
    task_11(
        p=p,
        x1=np.linspace(0., 1, 1000),
        dx=0.5
    )
    
    return

    print(measure_time(lambda: evaluate_interval_bounds(
        p=p, x1=np.linspace(0, 1, 5000000), dx=0.5)))
    print(measure_time(lambda: evaluate_vectorized_intervals(
        p=p, x1=np.linspace(0, 1, 5000000), dx=0.5)))


if __name__ == "__main__":
    main()