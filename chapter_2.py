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
    test_half_adder()
    test_full_adder()


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


def half_adder(A: int, B: int) -> tuple:
    sum = A ^ B
    carry = A & B
    return sum, carry


def test_half_adder():
    for A in [0, 1]:
        for B in [0, 1]:
            sum_result, carry_result = half_adder(A, B)
            print(f"""Half adder for A={A}, B={B}:\
Sum={sum_result}, Carry={carry_result}""")


def full_adder(A: int, B: int, Cin: int):
    sum = A ^ B ^ Cin
    carry_out = (A & B) | (B & Cin) | (A & Cin)
    return sum, carry_out


def test_full_adder():
    for A in [0, 1]:
        for B in [0, 1]:
            for C in [0, 1]:
                sum_result, carry_result = full_adder(A, B, C)
                print(f"""Full adder for A={A}, B={B}, C={C}:\
Sum={sum_result}, Carry={carry_result}""")


if __name__ == "__main__":
    main()
