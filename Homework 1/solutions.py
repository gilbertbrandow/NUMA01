import matplotlib.pyplot as pp
import math as m
from typing import Callable
import numpy as np

# Task 1
def approx_ln(x : float, n : int) -> float:
	a = (1+x)/2
	g = m.sqrt(x)

	for i in range(n):
		a = (a+g)/2
		g = m.sqrt(a*g)

	return (x-1)/a

def task_2():
	fig, axs = pp.subplots(2, 4)
	xv = np.linspace(0.1, 100, 50)
	nv = [1, 2, 5, 10]

	fig.set_figwidth(17)
	fig.set_figheight(10)

	for i in range(4):
		axs[0, i].plot(xv, [m.log(x) for x in xv])
		axs[0, i].plot(xv, [approx_ln(x, nv[i]) for x in xv])
		axs[0, i].set_title(f'ln and approx_ln, n={nv[i]}')
		axs[1,i].plot(xv, [abs(approx_ln(x, nv[i])-m.log(x)) for x in xv])
		axs[1, i].set_title(f'error, n={nv[i]}')

def task_3():
	pp.figure()
	pp.plot(range(50), [abs(m.log(1.41)-approx_ln(1.41, n)) for n in range(50)])
	pp.suptitle('Error of approx_ln(1.41, n) vs n')
	pp.xlabel("n")
	pp.xlabel("error")

# Task 4
def fast_approx_ln(x : float, n : int) -> float:
	a = (1+x)/2
	g = m.sqrt(x)

	dp = a # Either I am stupid, or the formula is wrong. What do I put here?? This seems to work...
	for i in range(1,n):
		d = a
		for k in range(1,i+1):
			d = (d-4**(-k)*dp)/(1-4**(-k))
		dp = d
		a = (a+g)/2
		g = m.sqrt(a*g)
	return (x-1)/d

# Task 5
def task_5():
	pp.figure()
	xv = np.linspace(0.1, 100, 50)
	for n in range(2, 7):
		pp.plot(xv, [abs(fast_approx_ln(x,n) - m.log(x)) for x in xv], label = f'n={n}')
	pp.xlabel("x")
	pp.xlabel("error")
	pp.suptitle('Error of fast_approx_ln for various n')
	pp.legend()

task_2()
task_3()
task_5()
pp.show()