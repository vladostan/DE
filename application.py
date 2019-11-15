# GUI Application showing different numerical methods for approximating solutions of differential equation(f'(x) = sol.function(x, y)).
# @rufusnufus, October 2019

import solution as sol
from matplotlib import pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np

#---------------------------------FIELD FOR TEXT---------------------------------#

class Field(object):
    def __init__(self, val, dims, str):
        self.val = val
        self.dims = dims
        self.str = str

    def draw(self):
        self.axes = plt.axes(self.dims)
        self.type = mwidgets.TextBox(self.axes, self.str + ':', initial = str(self.val))
        self.type.on_submit(self.on)

    def on(self, text):
        self.val = eval(text)

#-------------------------------GUI & CALCULATIONS-------------------------------#

class Application(object):
    def __init__(self, solution):
        self.solution = solution
        for s in self.solution:
            if isinstance(s, sol.Exact):
                self.exact = s

        self.figure = plt.figure(figsize=(9, 8), num="Differential Equations Assignment")

        self.x0_field = Field(self.exact.x_0, [0.06, 0.86, 0.1, 0.04], "x_0")
        self.x0_field.draw()
        self.y0_field = Field(self.exact.y_0, [0.06, 0.80, 0.1, 0.04], "y_0")
        self.y0_field.draw()
        self.xf_field = Field(self.exact.x_f, [0.06, 0.74, 0.1, 0.04], "X")
        self.xf_field.draw()
        self.n_field = Field(self.exact.n, [0.06, 0.68, 0.1, 0.04], "n")
        self.n_field.draw()
        self.n_min_field = Field(self.exact.n_min, [0.06, 0.62, 0.1, 0.04], "min_n")
        self.n_min_field.draw()
        self.n_max_field = Field(self.exact.n_max, [0.06, 0.56, 0.1, 0.04], "max_n")
        self.n_max_field.draw()
        self.button = mwidgets.Button(plt.axes([0.06, 0.5, 0.1, 0.03]), "Recalculate")
        self.button.on_clicked(self.on)  

        self.axes = self.figure.add_subplot(3, 1, 1)
        self.axes.set_title("Solutions of IVP")
        self.axes.set_xlabel("Value of x")
        self.axes.set_ylabel("Value of y")
        self.axes.grid(True)

        self.rescale(self.axes, self.x0_field.val, self.xf_field.val)

        for i in self.solution:
            i.graph(self.axes)
            i.calculate_error(self.exact.y)
            #i.calculate_max_error()
        plt.legend(loc = "upper left")

        # ERRORS

        self.axes_errors = self.figure.add_subplot(3, 1, 2)
        self.axes_errors.set_title("Errors of Numerical Methods")
        self.axes_errors.set_xlabel("Value of x")
        self.axes_errors.set_ylabel("Value of y")
        self.axes_errors.grid(True)

        self.rescale(self.axes_errors, self.x0_field.val, self.xf_field.val)

        for i in self.solution:
            if not isinstance(i, sol.Exact):
                i.graph_error(self.axes_errors)
        plt.legend(loc = "upper left")

        self.axes_max_errors = self.figure.add_subplot(3, 1, 3)
        self.axes_max_errors.set_title("Max errors of Numerical Methods")
        self.axes_max_errors.set_xlabel("Value of n")
        self.axes_max_errors.set_ylabel("Value of Total error")
        self.axes_max_errors.grid(True)

        self.rescale(self.axes_max_errors, self.n_min_field.val, self.n_max_field.val)

        n_values = np.linspace(self.n_min_field.val, self.n_max_field.val, self.n_max_field.val - self.n_min_field.val + 1)
        e_total_errors = np.zeros([self.n_max_field.val - self.n_min_field.val + 1])
        i_total_errors = np.zeros([self.n_max_field.val - self.n_min_field.val + 1])
        r_total_errors = np.zeros([self.n_max_field.val - self.n_min_field.val + 1])
        for i in range(self.n_min_field.val, self.n_max_field.val + 1):
            exact = sol.Exact(sol.function, self.x0_field.val, self.y0_field.val, self.xf_field.val, i)
            euler = sol.Euler(sol.function, self.x0_field.val, self.y0_field.val, self.xf_field.val, i)
            i_euler = sol.Improved_Euler(sol.function, self.x0_field.val, self.y0_field.val, self.xf_field.val, i)
            r_k = sol.Runge_Kutta(sol.function, self.x0_field.val, self.y0_field.val, self.xf_field.val, i)
            euler.calculate_error(exact.y)
            i_euler.calculate_error(exact.y)
            r_k.calculate_error(exact.y)
            e_total_errors[i - self.n_min_field.val] = max(np.amax(euler.err), abs(np.amin(euler.err)))
            i_total_errors[i - self.n_min_field.val] = max(np.amax(i_euler.err), abs(np.amin(i_euler.err)))
            r_total_errors[i - self.n_min_field.val] = max(np.amax(r_k.err), abs(np.amin(r_k.err)))
            del exact
            del euler
            del i_euler
            del r_k
        self.ax_max_err_0, = self.axes_max_errors.plot(n_values, e_total_errors, 'b-', label='Euler')
        self.ax_max_err_1, = self.axes_max_errors.plot(n_values, i_total_errors, 'g-', label='Improved Euler')
        self.ax_max_err_2, = self.axes_max_errors.plot(n_values, r_total_errors, 'r-', label='Runge-Kutta')

        plt.legend(loc = "upper left")

        self.lines = []
        for i in self.solution:
            self.lines.append(i.ax)

        self.lines_err = []
        for i in self.solution:
            if not isinstance(i, sol.Exact):
                self.lines_err.append(i.ax_err)

        self.lines_max_err = []
        self.lines_max_err.append(self.ax_max_err_0)
        self.lines_max_err.append(self.ax_max_err_1)
        self.lines_max_err.append(self.ax_max_err_2)

        # Make checkbuttons with all plotted lines with correct visibility
        rax = plt.axes([0.02, 0.33, 0.16, 0.15])
        self.labels = [str(line.get_label()) for line in self.lines]
        visibility = [line.get_visible() for line in self.lines]
        check = mwidgets.CheckButtons(rax, self.labels, visibility)
        check.on_clicked(self.tick)


        self.show()
            
    def recalculate(self):
        for s in self.solution:
            s.x_0 = self.x0_field.val
            s.y_0 = self.y0_field.val
            s.x_f = self.xf_field.val
            s.n = self.n_field.val
            s.n_min = self.n_min_field.val
            s.n_max = self.n_max_field.val
            s.recalculate(self.axes)
            if not isinstance(s, sol.Exact):
                s.calculate_error(self.exact.y)


        n_values = np.linspace(self.n_min_field.val, self.n_max_field.val, self.n_max_field.val - self.n_min_field.val + 1)
        e_total_errors = np.zeros([self.n_max_field.val - self.n_min_field.val + 1])
        i_total_errors = np.zeros([self.n_max_field.val - self.n_min_field.val + 1])
        r_total_errors = np.zeros([self.n_max_field.val - self.n_min_field.val + 1])
        for i in range(self.n_min_field.val, self.n_max_field.val + 1):
            exact = sol.Exact(sol.function, self.x0_field.val, self.y0_field.val, self.xf_field.val, i)
            euler = sol.Euler(sol.function, self.x0_field.val, self.y0_field.val, self.xf_field.val, i)
            i_euler = sol.Improved_Euler(sol.function, self.x0_field.val, self.y0_field.val, self.xf_field.val, i)
            r_k = sol.Runge_Kutta(sol.function, self.x0_field.val, self.y0_field.val, self.xf_field.val, i)
            euler.calculate_error(exact.y)
            i_euler.calculate_error(exact.y)
            r_k.calculate_error(exact.y)
            e_total_errors[i - self.n_min_field.val] = max(np.amax(euler.err), abs(np.amin(euler.err)))
            i_total_errors[i - self.n_min_field.val] = max(np.amax(i_euler.err), abs(np.amin(i_euler.err)))
            r_total_errors[i - self.n_min_field.val] = max(np.amax(r_k.err), abs(np.amin(r_k.err)))
            del exact
            del euler
            del i_euler
            del r_k

        self.ax_max_err_0.set_data(n_values, e_total_errors)
        self.ax_max_err_1.set_data(n_values, i_total_errors)
        self.ax_max_err_2.set_data(n_values, r_total_errors)

        self.rescale(self.axes, self.x0_field.val, self.xf_field.val)
        self.rescale(self.axes_errors, self.x0_field.val, self.xf_field.val)
        self.rescale(self.axes_max_errors, self.n_min_field.val, self.n_max_field.val)

    def rescale(self, axes, start, end):
        axes.set_xlim(start - 0.3, end + 0.3)
        axes.relim()
        axes.autoscale_view(True,True,True)

    def on(self, event):
        self.recalculate()

    def tick(self, label):
        index = self.labels.index(label)
        index_err = index - 1
        self.lines[index].set_visible(not self.lines[index].get_visible())
        if label != 'Exact':
            self.lines_err[index_err].set_visible(not self.lines_err[index_err].get_visible())
            self.lines_max_err[index_err].set_visible(not self.lines_max_err[index_err].get_visible())
        plt.draw()

    def show(self):
        plt.subplots_adjust(left=0.25, right=0.95, bottom=0.1, top=0.9, wspace=None, hspace=1.0)
        plt.show()

#--------------------------------RUNNING APPLICATION---------------------------------#

if __name__ == "__main__":
    a = sol.Exact(sol.function, 1, 1, 7, 40)
    b = sol.Euler(sol.function, 1, 1, 7, 40)
    c = sol.Improved_Euler(sol.function, 1, 1, 7, 40)
    d = sol.Runge_Kutta(sol.function, 1, 1, 7, 40)
    app = Application([a, b, c, d])
