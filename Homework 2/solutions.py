import matplotlib.pyplot as pp
import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Callable

class Interval():
    def __init__(self, a: float, b: Optional[float] = None) -> None:
        b = b or a
        self._a = float(min(a, b)) 
        self._b = float(max(a, b))

    @property
    def a(self) -> float:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    def __repr__(self) -> str:
        return f"[{self.a}, {self.b}]"

    def __add__(self, addend: Union['Interval', float, int]) -> 'Interval':
        if isinstance(addend, Interval):
            return Interval(self.a + addend.a, self.b + addend.b)
        elif isinstance(addend, (float, int)):
            return Interval(self.a + addend, self.b + addend)
        return NotImplemented

    def __radd__(self, addend: Union['Interval', float, int]) -> 'Interval':
        return self.__add__(addend)

    def __sub__(self, subtrahend: Union['Interval', float, int]) -> 'Interval':
        if isinstance(subtrahend, (float, int)):
            return Interval(self.a - subtrahend, self.b - subtrahend)
        elif not isinstance(subtrahend, Interval):
            return NotImplemented

        return Interval(self.a - subtrahend.b, self.b - subtrahend.a)

    def __rsub__(self, subtrahend: Union[float, int]) -> 'Interval':
        if isinstance(subtrahend, (float, int)):
            return Interval(subtrahend - self.b, subtrahend - self.a)
        return NotImplemented

    def __mul__(self, factor: Union[float, int, 'Interval']) -> 'Interval':
        if isinstance(factor, (float, int)):
            simple_products: tuple = (
                self.a * factor,
                self.b * factor
            )

            return Interval(min(simple_products), max(simple_products))
        elif not isinstance(factor, Interval):
            return NotImplemented

        products: tuple = (
            self.a * factor.a,
            self.a * factor.b,
            self.b * factor.a,
            self.b * factor.b
        )

        return Interval(
            min(products),
            max(products)
        )

    def __rmul__(self, factor: Union[float, int, 'Interval']) -> 'Interval':
        return self.__mul__(factor)

    def __truediv__(self, denominator: Union[float, int, 'Interval']) -> 'Interval':
        if isinstance(denominator, (float, int)):
            if denominator == 0:
                raise ZeroDivisionError("Cannot divide by 0")

            return Interval(self.a / denominator, self.b / denominator)
        elif not isinstance(denominator, Interval):
            return NotImplemented

        if denominator.a <= 0 <= denominator.b:
            raise ZeroDivisionError(
                f"Cannot divide by an interval that spans zero. Denominator: \
                    {denominator}"
            )

        quotients: tuple = (
            self.a / denominator.a,
            self.a / denominator.b,
            self.b / denominator.a,
            self.b / denominator.b
        )

        result: Interval = Interval(min(quotients), max(quotients))

        if result.b - result.a > 1e10:
            raise OverflowError(
                "Resulting interval is excessively large, indicating an approach toward infinity."
            )

        return result

    def __rtruediv__(self, other: Union[float, int]) -> 'Interval':
        if not isinstance(other, (float, int)):
            return NotImplemented

        if self.a <= 0 <= self.b:
            raise ZeroDivisionError(
                f"Cannot divide by an interval that spans zero. Interval \
                    {self}"
            )

        quotients: tuple = (
            other / self.a,
            other / self.b
        )

        return Interval(min(quotients), max(quotients))

    def __neg__(self) -> 'Interval':
        return Interval(-self.b, -self.a)

    def __contains__(self, x: Union[int, float]) -> bool:
        return self.a <= x <= self.b

    def __pow__(self, n: int) -> 'Interval':
        if not isinstance(n, int) or n < 1:
            raise ValueError("Exponent must be a positive integer.")

        if n % 2 == 1:
            return Interval(self.a ** n, self.b ** n)
        else:
            if self.a >= 0:
                return Interval(self.a ** n, self.b ** n)
            elif self.b < 0:
                return Interval(self.b ** n, self.a ** n)
            else:
                return Interval(0, max(self.a ** n, self.b ** n))

class VectorizedInterval():
    def __init__(self, a: npt.NDArray[np.float_], b: Optional[npt.NDArray[np.float_]] = None) -> None:
        if b is None:
            b = a
        self._a = np.asarray(a, dtype=np.float_)
        self._b = np.asarray(b, dtype=np.float_)
        self._a, self._b = np.minimum(self._a, self._b), np.maximum(self._a, self._b)

    @property
    def a(self) -> npt.NDArray[np.float_]:
        return self._a

    @property
    def b(self) -> npt.NDArray[np.float_]:
        return self._b

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

    y1: list[float] = [I.a for I in iv1]
    yu: list[float] = [I.b for I in iv2]

    pp.plot(x1, y1, label='y1')
    pp.plot(x1, yu, label='yu')

    pp.legend()
    pp.show()


def main() -> None:

    I: Interval = Interval(-1, 2)
    I1: Interval = Interval(1, 4)
    I2: Interval = Interval(-2, -1)
    I3: Interval = Interval(1)

    # task_4()
    
    task_10(
        p=lambda x: 3*x**3 - 2*x**2 - 5*x - 1,
        x1=np.linspace(0., 1, 1000),
        dx=0.5
    )


if __name__ == "__main__":
    main()
