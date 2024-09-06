from typing import Tuple
from numpy import *
import sys


def main() -> None:
    print(sys.version)
    task_8()
    print(f'Task 9: {task_9(x=1, y=2, z=4)}')
    print(f'Task 10a: {task_10(20, 6)}')
    print(f'Task 10b: {task_10(125, 11)}')

    if (task_12(x=2.3)):
        print("The expression is zero when x = 2.3")
    else:
        print("The expression is not zero when x = 2.3")

    print(f'Task 13: {task_13(2.2, 2)}')

    """
    Task 14: 
    The snippet calculates and prints the sum of the first 500 integers
    
           499
    sum = ∑ i
           i=0
    """


def task_8() -> None:
    sum_1: float = 0.25 + 0.2
    sum_2: float = 0.1 + 0.2
    x: int = 1
    y: float = 0.5

    print(sum_1)
    print(sum_2)
    print(x)
    print(y)


def task_9(x: int, y: int, z: int) -> float:
    sum: int = x + y
    product: int = sum * z
    difference: int = product - 10
    fraction: float = difference / 2
    return fraction


def task_10(numerator: int, denominator: int) -> Tuple[int, int]:
    quotient: int = numerator // denominator
    remainder: int = numerator % denominator
    return quotient, remainder


def task_11(x: float) -> float:
    """Native function for absolute value: abs()"""
    if x < 0:
        return -x
    return x


def task_12(x: float) -> bool:
    return x ** 2 + 0.25 * x - 5 == 0


def task_13(x: float, k: int) -> float:
    power: float = 1.0
    for _ in range(k):
        power *= x
    return power

    
if __name__ == "__main__":
    main()
