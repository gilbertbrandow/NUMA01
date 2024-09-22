from typing import Tuple, Callable
import numpy as np
from matplotlib.pyplot import *


def main() -> None:

    print(newton(
        lambda x: x**2 - 2,
        lambda x: 2*x,
        1.0,
        1e-8
    ))

    return


def newton(f: Callable[[float], float], fp: Callable[[float], float], x: float, TOL: float) -> Tuple[float, bool]:
    conv: bool = False

    for _ in range(0, 400):
        x_next: float = x - f(x)/fp(x)

        if np.abs(x_next - x) < TOL:
            conv = True
            break

        x = x_next

    return x, conv


def plot_complex_function() -> None:

    for i in range(1, 11):
        r: float = i / 10
        theta_vals = np.linspace(0, 2 * np.pi, 500)

        complex_vals = [complex_function(theta, r) for theta in theta_vals]
        x_vals = [z.real for z in complex_vals]
        y_vals = [z.imag for z in complex_vals]

        plot(x_vals, y_vals)

    xlabel('Re(z)')
    ylabel('Im(z)')
    legend()
    show()


def complex_function(theta: float, r: float) -> complex:
    return r*np.exp(1j*theta)


if __name__ == "__main__":
    main()
