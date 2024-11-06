import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Callable
from interval import Interval, VectorizedInterval, BaseInterval
import time


def task_4(I1: Interval, I2: Interval) -> None:
    sum: Interval = I1 + I2
    diff: Interval = I1 - I2
    product: Interval = I1 * I2
    quotient: Interval = I1 / I2

    print(sum)
    print(diff)
    print(product)
    print(quotient)


def task_10(
    p: Callable[[BaseInterval], BaseInterval],
    x1: npt.NDArray[np.floating],
    dx: float
) -> None:

    y1, yu = construct_10(p=p, x1=x1, dx=dx)

    plt.title("Task 10")
    plt.plot(x1, y1, label='y1')
    plt.plot(x1, yu, label='yu')

    plt.legend()
    plt.show()


def construct_10(
    p: Callable[[BaseInterval], BaseInterval],
    x1: npt.NDArray[np.floating],
    dx: float
) -> tuple[list[float], list[float]]:
    xu: npt.NDArray[np.floating] = x1 + dx
    iv: list[BaseInterval] = [p(Interval(l, u)) for l, u in zip(x1, xu)]

    y1: list[float] = [I.a for I in iv]
    yu: list[float] = [I.b for I in iv]

    return (y1, yu)


def measure_time(f: Callable) -> float:
    t_start: float = time.time()
    f()
    t_end: float = time.time()
    return t_end - t_start


def task_11(
    p: Callable[[BaseInterval], BaseInterval],
    x1: npt.NDArray[np.floating],
    dx: float
) -> None:
    vi = construct_11(p=p, x1=x1, dx=dx)

    plt.figure()
    plt.title("Task 11")
    plt.plot(x1, vi.a, label='y1')
    plt.plot(x1, vi.b, label='yu')
    plt.legend()
    plt.show()


def construct_11(
    p: Callable[[BaseInterval], BaseInterval],
    x1: npt.NDArray[np.floating],
    dx: float
) -> BaseInterval:
    xu: npt.NDArray[np.floating] = x1 + dx
    vi: BaseInterval = p(VectorizedInterval(x1, xu))
    return vi


def main() -> None:

    I: Interval = Interval(-1, 2)
    I1: Interval = Interval(1, 4)
    I2: Interval = Interval(-2, -1)
    I3: Interval = Interval(1)

    # task_4()

    p: Callable[[BaseInterval], BaseInterval] = lambda x: 3 * \
        x**3 - 2*x**2 - 5*x - 1

    task_10(
        p=p,
        x1=np.linspace(0., 1, 1000),
        dx=0.5
    )

    """ 
        Regarding 11.1
        We expect matrix * vector returns vector
    """
    
    # 11.2
    task_11(
        p=p,
        x1=np.linspace(0., 1, 1000),
        dx=0.5
    )

    # 11.3
    print(measure_time(lambda: construct_10(
        p=p, x1=np.linspace(0, 1, 5000000), dx=0.5)))
    print(measure_time(lambda: construct_11(
        p=p, x1=np.linspace(0, 1, 5000000), dx=0.5)))


if __name__ == "__main__":
    main()
