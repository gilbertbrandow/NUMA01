import matplotlib.pyplot as pp
import numpy as np

class Interval:
    def __init__ (self, a: float, b: float) -> None:
        self.a = a
        self.b = b
    
    def __repr__(self) -> str:
        return f"[{self.a}, {self.b}]"

    def __add__(self, other: 'Interval') -> 'Interval':
         return Interval(self.a + other.a, self.b + other.b)
        
def main() -> None:
    I: Interval = Interval(1, 2)
    I1: Interval = Interval (1 , 4 )
    I2: Interval = Interval ( -2 , - 1 )
    sum: Interval = I1 + I2 
    print(sum)

    
if __name__ == "__main__":
    main()