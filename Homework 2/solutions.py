import matplotlib.pyplot as pp
import numpy as np


class Interval:
    def __init__(
        self,
        a: float,
        b: float
    ) -> None:
        self.a = min(a, b)
        self.b = max(a, b)

    def __repr__(self) -> str:
        return f"[{self.a}, {self.b}]"

    def __add__(
        self,
        other: 'Interval'
    ) -> 'Interval':
        return Interval(self.a + other.a, self.b + other.b)

    def __sub__(
        self,
        other: 'Interval'
    ) -> 'Interval':
        return Interval(self.a - other.b, self.b - other.a)

    def __mul__(
        self,
        other: 'Interval'
    ) -> 'Interval':
        
        products: list = [
            self.a * other.a,
            self.a * other.b,
            self.b * other.a,
            self.b * other.b
        ]
        
        return Interval(
            min(products),
            max(products)
        )


def main() -> None:
    I: Interval = Interval(1, 2)
    I1: Interval = Interval(1, 4)
    I2: Interval = Interval(-2, - 1)
    sum: Interval = I1 + I2
    diff: Interval = I1 - I2
    product: Interval = I1 * I2
    print(sum)
    print(diff)
    print(product)


if __name__ == "__main__":
    main()
