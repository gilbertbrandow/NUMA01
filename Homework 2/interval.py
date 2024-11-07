from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union, Optional, Self
import numpy as np
import numpy.typing as npt

T = TypeVar('T', float, npt.NDArray[np.float_])

class BaseInterval(ABC, Generic[T]):
    """
    Abstract base class defining the interval interface.

    :param T: The type of the interval endpoints. Can be float or numpy ndarray.
    """

    @property
    @abstractmethod
    def a(self) -> T:
        """
        The lower bound of the interval.

        :return: The lower bound of the interval.
        """
        pass

    @property
    @abstractmethod
    def b(self) -> T:
        """
        The upper bound of the interval.

        :return: The upper bound of the interval.
        """
        pass

    @abstractmethod
    def _create_new_instance(self, a: T, b: T) -> Self:
        """
        Factory method to create a new instance of the subclass.

        :param a: The lower bound of the new interval.
        :param b: The upper bound of the new interval.
        :return: A new instance of the subclass.
        """
        pass

    def __add__(self, other: Union['BaseInterval[T]', T]) -> Self:
        """
         Add scalar or interval to interval. Implemented in the base class because
        the implementation is the same for Interval and VectorizedInterval.

        :param other: The other interval or scalar to add.
        :return: The sum of the intervals.
        """
        if isinstance(other, BaseInterval):
            return self._create_new_instance(self.a + other.a, self.b + other.b)
        elif isinstance(other, (float, int, np.ndarray)):
            return self._create_new_instance(self.a + other, self.b + other)
        else:
            return NotImplemented

    def __radd__(self, other: Union['BaseInterval[T]', T]) -> Self:
        """
        Add scalar or interval to interval (right-hand side).

        See __add__ for more details.

        :param other: The other interval or scalar to add.
        :return: The sum of the intervals.
        """
        return self.__add__(other)

    def __sub__(self, subtrahend: Union['BaseInterval', float, int]) -> Self:
        """
        Subtracts an interval or scalar from the current interval.

        Implemented in the base class for the same reason as __add__.

        :param subtrahend: The interval or scalar to subtract.
        :return: The difference of the intervals.
        """
        if isinstance(subtrahend, (float, int)):
            return self._create_new_instance(self.a - subtrahend, self.b - subtrahend)
        elif not isinstance(subtrahend, BaseInterval):
            return NotImplemented

        return self._create_new_instance(self.a - subtrahend.b, self.b - subtrahend.a)

    def __rsub__(self, subtrahend: Union[float, int]) -> Self:
        """
        Subtracts the current interval from a scalar (right-hand side).

        See __sub__ for more details.

        :param subtrahend: The scalar from which to subtract the current interval.
        :return: The difference of the intervals.
        """
        if isinstance(subtrahend, (float, int)):
            return self._create_new_instance(subtrahend - self.b, subtrahend - self.a)
        return NotImplemented

    def __neg__(self) -> Self:
        """
        Negates the interval.

        :return: The negation of the current interval.
        """
        return self._create_new_instance(-self.b, -self.a)

    @abstractmethod
    def __repr__(self):
        """
        Returns the string representation of the interval.

        :return: The string representation of the interval.
        """
        pass

    @abstractmethod
    def __mul__(self, factor: Union[float, int, Self]) -> Self:
        """
        Multiplies the interval with another interval or scalar.

        :param factor: The interval or scalar to multiply with.
        :return: The product of the intervals.
        """
        pass

    @abstractmethod
    def __rmul__(self, factor: Union[float, int, Self]) -> Self:
        """
        Multiplies the interval with another interval or scalar (right-hand side).

        :param factor: The interval or scalar to multiply with.
        :return: The product of the intervals.
        """
        pass

    @abstractmethod
    def __truediv__(self, denominator: Union[float, int, Self]) -> Self:
        """
        Divides the interval by another interval or scalar.

        :param denominator: The interval or scalar to divide by.
        :return: The quotient of the intervals.
        """
        pass

    @abstractmethod
    def __rtruediv__(self, numerator: Union[float, int, Self]) -> Self:
        """
        Divides another scalar by the current interval.

        :param numerator: The scalar numerator.
        :return: The quotient of the intervals.
        """
        pass

    @abstractmethod
    def __pow__(self, n: int) -> Self:
        """
        Raises the interval to an integer power.

        :param n: The exponent.
        :return: The interval raised to the power n.
        """
        pass


class Interval(BaseInterval[float]):
    """
    Represents a real-valued interval [a, b].

    :param a: The lower bound (or both bounds if b is None).
    :param b: The upper bound (optional). If None, interval is [a, a].
    """

    def __init__(self, a: float, b: Optional[float] = None) -> None:
        b = b or a
        self._a = float(min(a, b))
        self._b = float(max(a, b))

    @property
    def a(self) -> float:
        """
        The lower bound of the interval.

        :return: The lower bound of the interval.
        """
        return self._a

    @property
    def b(self) -> float:
        """
        The upper bound of the interval.

        :return: The upper bound of the interval.
        """
        return self._b

    def _create_new_instance(self, a: float, b: float) -> 'Interval':
        """
        Creates a new Interval instance.

        :param a: The lower bound.
        :param b: The upper bound.
        :return: A new Interval instance.
        """
        return Interval(a, b)

    def __repr__(self) -> str:
        """
        Returns the string representation of the interval.

        :return: The string representation of the interval.
        """
        return f"[{self.a}, {self.b}]"

    def __mul__(self, factor: Union[float, int, 'Interval']) -> 'Interval':
        """
        Multiplies the interval with another interval or scalar.

        :param factor: The interval or scalar to multiply with.
        :return: The product of the intervals.
        """
        if isinstance(factor, (float, int)):
            simple_products = (self.a * factor, self.b * factor)
            return Interval(min(simple_products), max(simple_products))
        elif not isinstance(factor, Interval):
            return NotImplemented

        products: tuple = (
            self.a * factor.a,
            self.a * factor.b,
            self.b * factor.a,
            self.b * factor.b
        )
        return Interval(min(products), max(products))

    def __rmul__(self, factor: Union[float, int, 'Interval']) -> 'Interval':
        """
        Multiplies the interval with another interval or scalar (right-hand side).

        See __mul__ for more details.

        :param factor: The interval or scalar to multiply with.
        :return: The product of the intervals.
        """
        return self.__mul__(factor)

    def __truediv__(self, denominator: Union[float, int, 'Interval']) -> 'Interval':
        """
        Divides the interval by another interval or scalar.

        :param denominator: The interval or scalar to divide by.
        :return: The quotient of the intervals.
        :raises ZeroDivisionError: If dividing by zero or an interval that spans zero.
        :raises OverflowError: If the resulting interval is excessively large.
        """
        if isinstance(denominator, (float, int)):
            if denominator == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
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
        result = Interval(min(quotients), max(quotients))

        if result.b - result.a > 1e10:
            raise OverflowError(
                "Resulting interval is excessively large, indicating an approach toward infinity."
            )

        return result

    def __rtruediv__(self, numerator: Union[float, int, 'Interval']) -> 'Interval':
        """
        Divides a scalar by the interval (right-hand side).

        :param numerator: The scalar numerator.
        :return: The quotient of the intervals.
        :raises ZeroDivisionError: If dividing by an interval that spans zero.
        """
        if not isinstance(numerator, (float, int)):
            return NotImplemented

        if self.a <= 0 <= self.b:
            raise ZeroDivisionError(
                f"Cannot divide by an interval that spans zero. Interval: {self}"
            )

        quotients: tuple = (
            numerator / self.a,
            numerator / self.b
        )
        return Interval(min(quotients), max(quotients))

    def __contains__(self, x: Union[int, float]) -> bool:
        """
        Checks if a value is within the interval.

        :param x: The value to check.
        :return: True if x is within the interval, False otherwise.
        """
        return self.a <= x <= self.b

    def __pow__(self, n: int) -> 'Interval':
        """
        Raises the interval to an integer power.

        :param n: The exponent (must be a positive integer).
        :return: The interval raised to the power n.
        :raises ValueError: If the exponent is not a positive integer.
        """
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
    """
    Represents an interval with vectorized endpoints (numpy arrays).

    :param a: The lower bounds (or both bounds if b is None).
    :param b: The upper bounds (optional). If None, interval is [a, a].
    """

    def __init__(self, a: npt.NDArray[np.float_], b: Optional[npt.NDArray[np.float_]] = None) -> None:
        if b is None:
            b = a
        self._a = np.asarray(a, dtype=np.float_)
        self._b = np.asarray(b, dtype=np.float_)
        self._a, self._b = np.minimum(
            self._a, self._b), np.maximum(self._a, self._b)

    @property
    def a(self) -> npt.NDArray[np.float_]:
        """
        The lower bounds of the interval.

        :return: The lower bounds as a numpy array.
        """
        return self._a

    @property
    def b(self) -> npt.NDArray[np.float_]:
        """
        The upper bounds of the interval.

        :return: The upper bounds as a numpy array.
        """
        return self._b

    def __repr__(self) -> str:
        """
        Returns the string representation of the interval.

        :return: The string representation of the interval.
        """
        return f"[{self.a}, {self.b}]"

    def _create_new_instance(self, a: np.ndarray, b: np.ndarray) -> 'VectorizedInterval':
        """
        Creates a new VectorizedInterval instance.

        :param a: The lower bounds as a numpy array.
        :param b: The upper bounds as a numpy array.
        :return: A new VectorizedInterval instance.
        """
        return VectorizedInterval(a, b)

    def __mul__(self, other: Union['VectorizedInterval', float, int, np.ndarray]) -> 'VectorizedInterval':
        """
        Multiplies the interval with another interval or scalar.

        :param other: The interval or scalar to multiply with.
        :return: The product of the intervals.
        """
        if isinstance(other, VectorizedInterval):
            products = np.array([
                self.a * other.a,
                self.a * other.b,
                self.b * other.a,
                self.b * other.b
            ])
            return VectorizedInterval(np.min(products, axis=0), np.max(products, axis=0))
        elif isinstance(other, (float, int, np.ndarray)):
            simple_products = np.array([self.a * other, self.b * other])
            return VectorizedInterval(np.min(simple_products, axis=0), np.max(simple_products, axis=0))
        else:
            return NotImplemented

    def __rmul__(self, other: Union['VectorizedInterval', float, int, np.ndarray]) -> 'VectorizedInterval':
        """
        Multiplies the interval with another interval or scalar (right-hand side).

        See __mul__ for more details.

        :param other: The interval or scalar to multiply with.
        :return: The product of the intervals.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Union['VectorizedInterval', float, int, np.ndarray]) -> 'VectorizedInterval':
        """
        Divides the interval by another interval or scalar.

        :param other: The interval or scalar to divide by.
        :return: The quotient of the intervals.
        :raises ZeroDivisionError: If dividing by zero or an interval that spans zero.
        """
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

    def __rtruediv__(self, numerator: Union['VectorizedInterval', float, int, np.ndarray]) -> 'VectorizedInterval':
        """
        Divides a scalar or array by the interval (right-hand side).

        :param numerator: The scalar or array numerator.
        :return: The quotient of the intervals.
        :raises ZeroDivisionError: If dividing by an interval that spans zero.
        """
        if not isinstance(numerator, (float, int, np.ndarray)):
            return NotImplemented

        if np.any((self.a <= 0) & (self.b >= 0)):
            raise ZeroDivisionError(
                f"Cannot divide by an interval that spans zero. Interval:\
                    {self}"
            )

        quotients = np.array([
            numerator / self.a,
            numerator / self.b
        ])
        return VectorizedInterval(np.min(quotients, axis=0), np.max(quotients, axis=0))

    def __pow__(self, n: int) -> 'VectorizedInterval':
        """
        Raises the interval to an integer power.

        :param n: The exponent (must be a positive integer).
        :return: The interval raised to the power n.
        :raises ValueError: If the exponent is not a positive integer.
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("Exponent must be a positive integer.")

        if n % 2 == 1:
            return VectorizedInterval(self.a ** n, self.b ** n)
        else:
            return VectorizedInterval(np.where(self.a >= 0, self.a ** n, 0), np.where(self.b >= 0, self.b ** n, np.maximum(self.a ** n, self.b ** n)))
        