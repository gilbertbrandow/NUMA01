from typing import Tuple, Callable
import numpy as np
from matplotlib.pyplot import *


def main() -> None:
    plot_complex_function()
    return

def plot_complex_function() -> None:
    
    for i in range(1, 11): 
        r: float = i / 10
        theta_vals = np.linspace(0, 2 * np.pi, 500)
        
        complex_vals = [complex_function(theta, r) for theta in theta_vals]
        x_vals = [z.real for z in complex_vals] 
        y_vals = [z.imag for z in complex_vals] 
        
        plot(x_vals, y_vals)
    
    xlabel('Re(z)')
    ylabel('Im(z)')
    legend()
    show()

def complex_function(theta: float, r: float) -> complex:
    return r*np.exp(1j*theta)

if __name__ == "__main__":
    main()
