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
        addend: 'Interval'
    ) -> 'Interval':
        return Interval(self.a + addend.a, self.b + addend.b)

    def __sub__(
        self,
        subtrahend: 'Interval'
    ) -> 'Interval':
        return Interval(self.a - subtrahend.b, self.b - subtrahend.a)

    def __mul__(
        self,
        factor: 'Interval'
    ) -> 'Interval':

        products: list = [
            self.a * factor.a,
            self.a * factor.b,
            self.b * factor.a,
            self.b * factor.b
        ]

        return Interval(
            min(products),
            max(products)
        )

    def __truediv__(
            self,
            denominator: 'Interval'
    ) -> 'Interval':
        
        if denominator.a <= 0 <= denominator.b:
            raise ValueError(f"Cannot divide by an interval that spans zero. Denominator: {denominator}")

        quotients: list = [
            self.a / denominator.a,
            self.a / denominator.b,
            self.b / denominator.a,
            self.b / denominator.b
        ]

        return Interval(min(quotients), max(quotients))


def main() -> None:
    I: Interval = Interval(-1, 2)
    I1: Interval = Interval(1, 4)
    I2: Interval = Interval(-2, -1)
    
    # Task 3: The four basic arithmetic operations
    sum: Interval = I1 + I2
    diff: Interval = I1 - I2
    product: Interval = I1 * I2
    quotient: Interval = I1 / I2
    
    print(sum)
    print(diff)
    print(product)
    print(quotient)


if __name__ == "__main__":
    main()
