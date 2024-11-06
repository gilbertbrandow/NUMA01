import matplotlib.pyplot as pp
import numpy as np
import numpy.typing as npt
from typing import Callable
from interval import Interval, VectorizedInterval
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
        p: Callable[[Interval], Interval],
        x1: npt.NDArray[np.floating],
        dx: float
) -> None:
    xu: npt.NDArray[np.floating] = x1 + dx

    iv1: list[Interval] = [Interval(l, u) for l, u in zip(x1, xu)]
    iv2: list[Interval] = [p(I) for I in iv1]

    y1: list[float] = [I.a for I in iv2]
    yu: list[float] = [I.b for I in iv2]

    pp.plot(x1, y1, label='y1')
    pp.plot(x1, yu, label='yu')

    pp.legend()
    pp.show()

def measure_time(f: Callable) -> float:
    t_start: float = time.time()
    f()
    t_end: float = time.time()
    return t_end - t_start

def task_11(
    p: Callable[[VectorizedInterval], VectorizedInterval],
    x1: npt.NDArray[np.floating],
    dx: float
) -> None:
    xu: npt.NDArray[np.floating] = x1 + dx
    vi: VectorizedInterval = p(VectorizedInterval(x1, xu))

    pp.figure()
    pp.plot(x1, vi.a, label='y1')
    pp.plot(x1, vi.b, label='yu')
    pp.legend()
    pp.show()

def main() -> None:

    I: Interval = Interval(-1, 2)
    I1: Interval = Interval(1, 4)
    I2: Interval = Interval(-2, -1)
    I3: Interval = Interval(1)

    # task_4()

    p: Callable[[BaseInterval], BaseInterval]=lambda x: 3*x**3 - 2*x**2 - 5*x - 1
    
    task_10(
        p=p
        x1=np.linspace(0., 1, 1000),
        dx=0.5
    )
    
    # 11.2
    task_11(
        p=p,
        x1=np.linspace(0., 1, 1000),
        dx=0.5
    )

    # 11.3
    print(measure_time(lambda: task_10(p=p, x1=linspace(0,1,5000000), dx=0.5)))
    print(measure_time(lambda: task_11(p=p, x1=linspace(0,1,5000000), dx=0.5)))

if __name__ == "__main__":
    main()
