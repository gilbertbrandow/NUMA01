import matplotlib.pyplot as pp
import numpy as np
from typing import List


def approx_ln(x: float, n: int) -> float:
    if x < 0:
        raise Exception("x must be greater than 0.")

    a: float = (1+x)/2
    g: float = np.sqrt(x)

    for _ in range(n):
        a = (a+g)/2
        g = np.sqrt(a*g)

    return (x-1)/a


def task_2() -> None:
    fig, axs = pp.subplots(2, 4)
    xv = np.linspace(0.1, 100, 50)
    nv = [1, 2, 5, 10]

    fig.set_figwidth(17)
    fig.set_figheight(10)

    for i in range(4):
        axs[0, i].plot(xv, [np.log(x) for x in xv])
        axs[0, i].plot(xv, [approx_ln(x, nv[i]) for x in xv])
        axs[0, i].set_title(f'ln and approx_ln, n={nv[i]}')
        axs[1, i].plot(xv, [abs(approx_ln(x, nv[i])-np.log(x)) for x in xv])
        axs[1, i].set_title(f'error, n={nv[i]}')


def task_3():
    pp.figure()
    pp.plot(range(50), [abs(np.log(1.41)-approx_ln(1.41, n))
            for n in range(50)])
    pp.suptitle('Error of approx_ln(1.41, n) vs n')
    pp.xlabel("n")
    pp.ylabel("error")



def fast_approx_ln(x: float, n: int) -> float:
    if x < 0:
        raise Exception("x must be greater than 0.")
    
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
    pp.figure()
    xv = np.linspace(0.1, 20, 50)
    for n in range(2, 7):
        pp.plot(xv, [abs(fast_approx_ln(x, n) - np.log(x)) for x in xv], label=f'iteration {n}')
    
    pp.xlabel("x")
    pp.ylabel("error")
    pp.yscale('log')
    pp.suptitle('Error behavior of the accelerated Carlsson method for the log')
    pp.legend()
    pp.grid(True)


def main() -> None:
    task_2()
    task_3()
    task_5()
    pp.show()


if __name__ == "__main__":
    main()
