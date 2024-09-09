from typing import Tuple, Callable
import numpy as np
import sys


def main() -> None:
    z = 3.25+5.2j
    print(z)
    print(np.conjugate(z))
    print(z.imag)
    print(np.conjugate(z).imag)
    print(z * np.conjugate(z))
    print(exercise_1())
    print(exercise_2(20))
    print(exercise_3(20))
    exercise_4()
    print(implication(True, True))


def exercise_1() -> bool:
    def f(x: float) -> float:
        return x**2 - 0.25*x + 5

    return not f(2.3)


def exercise_2(number_of_checks: int) -> bool:
    for _ in range(number_of_checks):
        if not check_de_moivres_formula(np.random.randint(1, 100), np.random.uniform(0, 100)):
            return False
    return True


def check_de_moivres_formula(n: int, x: float) -> bool:
    lhs = (np.cos(x) + np.sin(x)*1j)**n
    rhs = np.cos(n*x) + np.sin(n*x)*1j

    # Experienced issues with floating point precision, so resorted to the np.isclose function
    return np.isclose(lhs.real, rhs.real) and np.isclose(lhs.imag, rhs.imag)


def exercise_3(number_of_checks: int) -> bool:
    for _ in range(number_of_checks):
        if not check_de_moivres_formula(np.random.randint(1, 100), np.random.uniform(0, 100)):
            return False
    return True


def check_eulers_formula(x: float) -> bool:
    lhs = np.e**(x*1j)
    rhs = np.cos(x) + np.sin(x)*1j

    return np.isclose(lhs.real, rhs.real) and np.isclose(lhs.imag, rhs.imag)


def exercise_4(u: int = 1, uold: float = 10.0) -> None:
    for _ in range(2000):
        if not np.abs(u-uold) > 1.e-8:
            print('Convergence')
            break
        uold = u
        u = 2*u
    else:
        print('No convergence')


def implication(A: bool, B: bool) -> bool:
    return not A or (A and B)


if __name__ == "__main__":
    main()
