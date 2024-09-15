import numpy as np
from matplotlib.pyplot import *

def main()->None:
    x = 0.5
    for i in range(200):
        x = x**2
        
    print(f'The result after { i + 1 } iterations is { x }')
    
    #first_plot()
    print(task_3(x=2.3))
    L3: list = task_4()
    L4: list = task_5(L3=L3)
    L5: list = task_6(L3=L3, L4=L4)
    task_8()
    
    xplot: list = task_9()
    yplot: list = task_10(xplot=xplot)
    #task_11(xplot=yplot, yplot=yplot)
    
    print(task_12())
    return

def first_plot()->None: 
    x_vals = [.2*n for n in range(20)]
    y1 = [np.sin(.3*x) for x in x_vals]
    y2 = [np.sin(2*x) for x in x_vals]
    plot(x_vals ,y1, label='sin(0.3*x)')
    plot(x_vals, y2, label='sin(2*x)')
    xlabel('x')
    ylabel('sin...')
    legend()
    show()
    
def task_3(x: float)->bool: 
    return x**2 + 0.25*x - 5 == 0

def task_4()->list:
    L: list = [1, 2]
    L3: list = 3*L
    
    return L3
    
def task_5(L3: list)->list: 
    L4: list = [k**2 for k in L3] # creates a new list at L4 but raises the values of all L3 to the power of 2
    return L4
    
def task_6(L3: list, L4: list)->list: 
    L5: list = L3 + L4
    return L5

def task_7()->list: 
    L6: list = [n/99 for n in range(100)]
    return L6

def task_8()->None: 
    # Sum all numbers from 0 to 500
    s = 0
    for i in range(0, 500): 
        s = s + i
    print(s)
    
    ss: list = [0]
    for i in range(1, 500): 
        ss.append(ss[i-1]+i)
    # i has the same value, 499
    print(ss)

def task_9()->list: 
    xplot = []
    for i in range(100):
        xplot.append(i / 99)
    return xplot

def task_10(xplot: list)->list: 
    yplot = [np.arctan(x) for x in xplot]
    return yplot

def task_11(xplot: list, yplot: list)->None:
    plot(xplot, yplot, label='arctan(x)')
    xlim([0, 1])
    xlabel('x')
    ylabel('arctan')
    legend()
    show()
    
def task_12()->float:
    return sum(1 / np.sqrt(i) for i in range(1, 201))
        
    
if __name__ == "__main__":
    main()