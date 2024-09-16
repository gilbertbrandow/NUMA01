from numpy import *
from matplotlib.pyplot import *

n: range = range(3, 1001)
h: float = 1 / 1000
a: float = -0.5


def main(n: range, h: float, a: float) -> None:
    u: list = [np.exp(0), np.exp(a*h), np.exp(2*a*h)]

    for index in n:
        u.append(calculate_by_index(index=index, u=u, h=h, a=a))

    td: list = [n*h for n in range(0, 1001)]

    # Plot the approximation u_n versus td
    figure(figsize=(10, 5))
    plot(td, u, label='Approximation (Recursion Formula)', color='blue')
    xlabel('x')
    ylabel('approximation')
    title('Plot of Approximation (td vs u)')
    legend()
    grid(True)
    show()

    # Exact values e^(a * tn)
    exact_values = [np.exp(a * t) for t in td]

    # Approximation and Exact solution
    figure(figsize=(10, 5))
    plot(td, u, label='Approximation (Recursion Formula)', color='blue')
    plot(td, exact_values, label='Exact Solution (e^(a * tn))', color='green', linestyle='--')
    xlabel('x')
    ylabel('y')
    title('Approximation vs Exact Solution')
    legend()
    grid(True)
    show()

    # Calculate the difference |e^(a*tn) - u_n|
    differences = [np.abs(exact - approx) for exact, approx in zip(exact_values, u)]

    # Plot the differences |e^(a*tn) - u_n|
    figure(figsize=(10, 5))
    plot(td, differences, label='|Exact Solution - Approximation|', color='red')
    xlabel('x')
    ylabel('Absolute Difference')
    title('Plot of Absolute Differences |e^(a*tn) - u_n|')
    legend()
    grid(True)
    show()
    
    return


def calculate_by_index(index: int, u: list, h: float, a: float) -> float:
    n = index - 3
    return u[n+2] + h*a*(23/12*u[n+2] - 4/3*u[n+1] + 5/12*u[n])


if __name__ == "__main__":
    main(n=n, h=h, a=a)
