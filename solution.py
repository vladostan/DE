import numpy as np
import math
import matplotlib


#------------------------------------SOLUTIONS------------------------------------#

class Solution(object):
    def __init__(self, func, x_0, y_0, x_f, n):
        self.x_0 = x_0
        self.y_0 = y_0
        self.x_f = x_f
        self.n = n
        self.n_min = n - 10
        self.n_max = n + 10
        self.func = func
        self.h = (x_f - x_0) / (n)
        self.x = np.linspace(x_0, x_f, n + 1)
        self.y = np.zeros([n + 1])
        self.err = np.zeros([n + 1])
        self.y[0] = y_0
        self.ax = matplotlib.lines.Line2D([],[]) 
        self.ax_err = matplotlib.lines.Line2D([],[]) 
        self.solve()

    def recalculate(self, axes):
        self.x = np.linspace(self.x_0, self.x_f, self.n + 1)
        self.y = np.zeros([self.n + 1])
        self.h = (self.x_f - self.x_0) / (self.n)
        self.y[0] = self.y_0
        self.solve()
        self.ax.set_data(self.x, self.y)

    def calculate_error(self, y_exact):
        self.err = y_exact - self.y
        self.ax_err.set_data(self.x, self.err)

    def solve(self):
        pass

    def graph(self, axes):
        pass

class Exact(Solution):
    def get_const(self):
        return (self.y_0 + 2 * self.x_0 - 1) / math.exp(self.x_0)

    def solve(self):
        const = self.get_const()
        for i in range(1, self.n + 1):
            self.y[i] = (-2) * self.x[i] + 1 + 2 * math.exp(self.x[i] - 1)

    def graph(self, axes):
        self.ax, = axes.plot(self.x, self.y, 'o-', label='Exact', markersize =3)

    
class Euler(Solution):
    def solve(self):
        for i in range(1, self.n + 1):
            self.y[i] = self.h * self.func(self.x[i - 1], self.y[i - 1]) + self.y[i - 1]

    def graph(self, axes):
        self.ax, = axes.plot(self.x, self.y, 'b-', label='Euler')

    def graph_error(self, axes):
        self.ax_err, = axes.plot(self.x, self.err, 'b-', label='Euler')
    
class Improved_Euler(Solution):
    def solve(self):
        for i in range(1, self.n + 1):
            self.y[i] = self.h * self.func(self.x[i - 1], self.y[i - 1]) + self.y[i - 1]
            self.y[i] = (self.func(self.x[i - 1], self.y[i - 1]) + self.func(self.x[i], self.y[i]))
            self.y[i] = self.h * self.y[i] / 2 + self.y[i - 1]
    
    def graph(self, axes):
        self.ax, = axes.plot(self.x, self.y, 'g-', label='Improved Euler')

    def graph_error(self, axes):
        self.ax_err, = axes.plot(self.x, self.err, 'g-', label='Improved Euler')

class Runge_Kutta(Solution):
    def solve(self):
        k = np.zeros([4])
        for i in range(1, self.n + 1):
            k[0] = self.func(self.x[i - 1], self.y[i - 1])
            k[1] = self.func(self.x[i - 1] + self.h / 2, self.y[i - 1] + self.h * k[0] / 2)
            k[2] = self.func(self.x[i - 1] + self.h / 2, self.y[i - 1] + self.h * k[1] / 2)
            k[3] = self.func(self.x[i - 1] + self.h, self.y[i - 1] + self.h * k[2])
            self.y[i] = (self.h / 6) * (k[0] + 2 * k[1] + 2 * k[2] + k[3]) + self.y[i - 1]

    def graph(self, axes):
        self.ax, = axes.plot(self.x, self.y, 'r-', label='Runge-Kutta')

    def graph_error(self, axes):
        self.ax_err, = axes.plot(self.x, self.err, 'r-', label='Runge-Kutta')

#------------------------------------FUNCTION------------------------------------#

def function(x, y):
    return 2 * x + y - 3