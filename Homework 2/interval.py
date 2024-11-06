from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union, Optional, Self
import numpy as np
import numpy.typing as npt

T = TypeVar('T', float, npt.NDArray[np.float_])


class BaseInterval(ABC, Generic[T]):
    """Abstract base class defining the interval interface."""

    @property
    @abstractmethod
    def a(self) -> T:
        pass

    @property
    @abstractmethod
    def b(self) -> T:
        pass

    @abstractmethod
    def _create_new_instance(self, a: T, b: T) -> Self:
        """Factory method to create a new instance of the subclass."""
        pass

    def __add__(self, other: Union['BaseInterval[T]', T]) -> Self:
        """
        Adds two intervals together. Implemented in the base class because
        the implementation is same for Interval and VectorizedInterval.
        
        :param self: The current interval
        :param other: The other interval
        
        :returns: The sum of the intervals
        """
        
        if isinstance(other, BaseInterval):
            return self._create_new_instance(self.a + other.a, self.b + other.b)
        elif isinstance(other, (float, int, np.ndarray)):
            return self._create_new_instance(self.a + other, self.b + other)
        else:
            return NotImplemented

    def __radd__(self, other: Union['BaseInterval[T]', T]) -> Self:
        """
        Adds two intervals together, see __add__.

        :param other: The other interval
        :param self: The current interval
        
        :returns: The sum of the intervals
        """
        return self.__add__(other)

    def __sub__(self, subtrahend: Union['BaseInterval', float, int]) -> Self:
        """
        Subtracts two intervals. Implemented in the base class for
        the same reason as __add__.

        :param self: The current interval
        :param other: The other interval
        
        :returns: The difference of the intervals
        """
        if isinstance(subtrahend, (float, int)):
            return self._create_new_instance(self.a - subtrahend, self.b - subtrahend)
        elif not isinstance(subtrahend, BaseInterval):
            return NotImplemented

        return self._create_new_instance(self.a - subtrahend.b, self.b - subtrahend.a)

    def __rsub__(self, subtrahend: Union[float, int]) -> Self:
        """
        Subtracts two intervals together, see __sub__.

        :param other: The other interval
        :param self: The current interval
        
        :returns: The difference of the intervals
        """
        if isinstance(subtrahend, (float, int)):
            return self._create_new_instance(subtrahend - self.b, subtrahend - self.a)
        return NotImplemented

    def __neg__(self) -> Self:
        """
        Negates the interval.

        :param self: The current interval

        :returns: The negation of the current interval
        """
        return self._create_new_instance(-self.b, -self.a)

    # Note: these methods have subclass-specific implementations, so they are abstract here
    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __mul__(self, factor: Union[float, int, Self]) -> Self:
        pass

    @abstractmethod
    def __rmul__(self, factor: Union[float, int, Self]) -> Self:
        pass

    @abstractmethod
    def __truediv__(self, denominator: Union[float, int]) -> Self:
        pass

    @abstractmethod
    def __rtruediv__(self, denominator: Union[float, int]) -> Self:
        pass

    @abstractmethod
    def __pow__(self, n: int) -> Self:
        pass


class Interval(BaseInterval[float]):
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

    def _create_new_instance(self, a: float, b: float) -> 'Interval':
        return Interval(a, b)

    def __repr__(self) -> str:
        return f"[{self.a}, {self.b}]"

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

    def __rtruediv__(self, denominator: Union[float, int]) -> 'Interval':
        if not isinstance(denominator, (float, int)):
            return NotImplemented

        if self.a <= 0 <= self.b:
            raise ZeroDivisionError(
                f"Cannot divide by an interval that spans zero. Interval \
                    {self}"
            )

        quotients: tuple = (
            denominator / self.a,
            denominator / self.b
        )

        return Interval(min(quotients), max(quotients))

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


class VectorizedInterval(BaseInterval[npt.NDArray[np.float_]]):
    def __init__(self, a: npt.NDArray[np.float_], b: Optional[npt.NDArray[np.float_]] = None) -> None:
        if b is None:
            b = a
        self._a = np.asarray(a, dtype=np.float_)
        self._b = np.asarray(b, dtype=np.float_)
        self._a, self._b = np.minimum(
            self._a, self._b), np.maximum(self._a, self._b)

    @property
    def a(self) -> npt.NDArray[np.float_]:
        return self._a

    @property
    def b(self) -> npt.NDArray[np.float_]:
        return self._b

    def __repr__(self) -> str:
        return f"[{self.a}, {self.b}]"

    def _create_new_instance(self, a: np.ndarray, b: np.ndarray) -> 'VectorizedInterval':
        return VectorizedInterval(a, b)

    def __mul__(self, other: Union['VectorizedInterval', float, int, np.ndarray]) -> 'VectorizedInterval':
        if isinstance(other, VectorizedInterval):
            products = np.array([
                self.a * other.a,
                self.a * other.b,
                self.b * other.a,
                self.b * other.b
            ])

            return VectorizedInterval(np.min(products, axis=0), np.max(products, axis=0))
        elif isinstance(other, (float, int, np.ndarray)):
            simple_products: npt.NDArray[np.float_] = np.array(
                [self.a * other, self.b * other])
            return VectorizedInterval(np.min(simple_products, axis=0), np.max(simple_products, axis=0))
        else:
            return NotImplemented

    def __rmul__(self, other: Union['VectorizedInterval', float, int, np.ndarray]) -> 'VectorizedInterval':
        return self.__mul__(other)

    def __truediv__(self, other: Union['VectorizedInterval', float, int, np.ndarray]) -> 'VectorizedInterval':
        if isinstance(other, VectorizedInterval):
            if np.any((other.a <= 0) & (other.b >= 0)):
                raise ZeroDivisionError(
                    "Cannot divide by an interval that spans zero.")

            quotients = np.array([
                self.a / other.a,
                self.a / other.b,
                self.b / other.a,
                self.b / other.b
            ])

            return VectorizedInterval(np.min(quotients, axis=0), np.max(quotients, axis=0))

        elif isinstance(other, (float, int, np.ndarray)):
            if np.any(other == 0):
                raise ZeroDivisionError("Cannot divide by zero.")
            return VectorizedInterval(self.a / other, self.b / other)
        else:
            return NotImplemented

    def __rtruediv__(self, numerator: Union[float, int, np.ndarray]) -> 'VectorizedInterval':
        if not isinstance(numerator, (float, int, np.ndarray)):
            return NotImplemented

        if np.any((self.a <= 0) & (self.b >= 0)):
            raise ZeroDivisionError(
                f"Cannot divide by an interval that spans zero. Interval \
                    {self}"
            )

        quotients = np.array([
            numerator / self.a,
            numerator / self.b
        ])

        return VectorizedInterval(np.min(quotients, axis=0), np.max(quotients, axis=0))

    def __pow__(self, n: int) -> 'VectorizedInterval':
        if not isinstance(n, int) or n < 1:
            raise ValueError("Exponent must be a positive integer.")

        if n % 2 == 1:
            return VectorizedInterval(self.a ** n, self.b ** n)
        else:
            return VectorizedInterval(np.where(self.a >= 0, self.a ** n, 0), np.where(self.b >= 0, self.b ** n, np.maximum(self.a ** n, self.b ** n)))
