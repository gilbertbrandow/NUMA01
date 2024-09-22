from typing import Tuple, Callable
import numpy as np
from matplotlib.pyplot import *


def main() -> None:
    
    print(bisection_method(lambda x: x**3 - x - 2, 1, 2))
    print(bisection_method(lambda x: np.arctan(x), -1, 2))
    
    f = lambda x: 3*x**2 - 5
    
    try:
        root_1 = bisection_method(f, -0.5, 0.6)
        print(f"Root in the interval [-0.5, 0.6]: {root_1}")
    except ValueError as e:
        print(e)

    try:
        root_2 = bisection_method(f, -1.5, -0.4)
        print(f"Root in the interval [-1.5, -0.4]: {root_2}")
    except ValueError as e:
        print(e)
    
    return


def try_accessing_list() -> None:
    L = [0, 1, 2, 1, 0, -1, -2, -1, 0]

    print(L[0])
    print(L[- 1])
    print(L[: - 1])
    print(L + L[1: - 1] + L)

    print("in place operations: ")
    L[2: 2] = [- 3]
    print(L)
    L[3: 4] = []
    print(L)
    L[2: 5] = [- 5]
    print(L)
    return


def task_4() -> list[list[int]]:
    distance: list = [
        [0, 20, 30, 40],
        [20, 0, 50, 60],
        [30, 50, 0, 70],
        [40, 60, 70, 0]
    ]
    
    reddistance: list = []
    
    # Using for loops
    for i in range(1, len(distance)):
        reddistance.append(distance[i][:i])

    # List comprehenstion
    reddistance =  [distance[i][:i] for i in range(1, len(distance))]
    
    return reddistance

def symmetric_difference(A: set, B: set) -> set: 
    return (A - B) | (B - A)

def bisection_method(function: Callable[[float], float], a: float, b: float, tolerance: float = 1e-8) -> float:
    """
    Recursively approximates the root using the bisection method

    Args:
        function (Callable[[float], float]): _description_
        a (float): _description_
        b (float): _description_
        tolerance (float, optional): _description_. Defaults to 1e-8.

    Raises:
        ValueError: _description_

    Returns:
        float: _description_
    """
    if not function(a) * function(b) < 0: 
        raise ValueError(f'There is no sign change between the given points {a} and {b}')
    
    if np.abs(a - b) < tolerance: 
        return (a + b) / 2
    
    if function(a) * function((a+b)/2) < 0: 
        return bisection_method(function=function, a=a, b=(a+b)/2, tolerance=tolerance)
    else: 
        return bisection_method(function=function, a=(a+b)/2, b=b, tolerance=tolerance)

if __name__ == "__main__":
    main()
