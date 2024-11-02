import matplotlib.pyplot as pp
import numpy as np
from typing import List


def approx_ln(x: float, n: int) -> float:
    """
    Approximates the natural logarithm of a number using the Carlson method.

    :param x: The number for which to approximate the natural logarithm. Must be positive.
    :param n: The number of iterations to refine the approximation.
    :return: The approximation of ln(x).
    :raises ValueError: if x is not greater than 0.
    """
    if x <= 0:
        raise ValueError("x must be greater than 0.")

    a: float = (1+x)/2
    g: float = np.sqrt(x)

    for _ in range(n):
        a = (a+g)/2
        g = np.sqrt(a*g)

    return (x-1)/a


def task_2() -> None:
    """
    Generates plots to compare the accuracy of the approx_ln function against the actual natural logarithm function.

    The function creates two rows of subplots: the first row compares ln(x) with approx_ln(x), and
    the second row shows the error between them. The comparisons are made for different iteration counts n.
    """
    fig, axs = pp.subplots(2, 4)
    xv = np.linspace(0.1, 100, 50)
    nv: List = [1, 2, 5, 10]

    fig.set_figwidth(17)
    fig.set_figheight(10)

    for i in range(4):
        axs[0, i].plot(xv, [np.log(x) for x in xv], label='ln(x)')
        axs[0, i].plot(xv, [approx_ln(x, nv[i]) for x in xv], label='approx_ln(x)')
        axs[0, i].set_title(f'n={nv[i]}')
        axs[0, i].legend()

        axs[1, i].plot(xv, [abs(approx_ln(x, nv[i])-np.log(x)) for x in xv], label='error ln(x) and approx_ln(x)')
        axs[1, i].set_title(f'n={nv[i]}')
        axs[1, i].legend()


def task_3():
    """
    Plots the error of approx_ln(1.41, n) as a function of n.

    This function shows how the error decreases as the number of iterations (n) increases for a fixed input x = 1.41.
    """
    pp.figure()
    pp.plot(range(50), [abs(np.log(1.41)-approx_ln(1.41, n))
            for n in range(50)])
    pp.suptitle('Error of approx_ln(1.41, n) vs n')
    pp.xlabel("n")
    pp.ylabel("error")


def fast_approx_ln(x: float, n: int) -> float:
    """
    Fast approximation of the natural logarithm.

    This method improves the accuracy of the approximation by using a two-dimensional array
    to store intermediate results based on the AGM method, applying a technique similar to Richardson extrapolation.

    :param x: The number for which to approximate the natural logarithm. Must be positive.
    :param n: The number of iterations to refine the approximation.
    :return: The approximation of ln(x).
    :raises ValueError: if x is not greater than 0.
    """
    if x <= 0:
        raise ValueError("x must be greater than 0.")
    
    d: List[List[float]] = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
    a: float = (1 + x) / 2
    g: float = np.sqrt(x)
    d[0][0] = a
    
    for i in range(1, n + 1):
        a = (a + g) / 2
        g = np.sqrt(a * g)
        d[0][i] = a
      
    for i in range(1, n + 1):
        for k in range(1, i + 1):
            d[k][i] = (d[k - 1][i] - 4 ** (-k) * d[k - 1][i - 1]) / (1 - 4 ** (-k))
    
    return (x - 1) / d[n][n]


def task_5():
    """
    Plots the error behavior of the fast_approx_ln method for various iteration counts.

    The plot shows the error between fast_approx_ln(x, n) and ln(x) as x varies, with different iteration counts (n).
    The y-axis is set to a logarithmic scale to show how the error decreases with more iterations.
    """
    pp.figure()
    xv = np.linspace(0.1, 20, 200)
    
    for n in range(2, 7):
        pp.scatter(xv, [abs(fast_approx_ln(x, n) - np.log(x)) for x in xv], label=f'iteration {n}')
    
    pp.xlabel("x")
    pp.ylabel("error")
    pp.yscale('log')
    pp.suptitle('Error behavior of the accelerated Carlsson method for the log')
    pp.legend()
    pp.grid(True)


def main() -> None:
    """
    The main function that calls other tasks and displays the plots.

    task_2: Compares the approximation of ln(x) with different iteration counts.
    task_3: Shows the error of approx_ln for a fixed x over a range of iteration counts.
    task_5: Displays the error behavior of fast_approx_ln for different iteration counts.
    """
    task_2()
    task_3()
    task_5()
    pp.show()


if __name__ == "__main__":
    main()
