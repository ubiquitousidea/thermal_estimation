import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap, Normalize
from matplotlib.pyplot import scatter
from numpy import float128 as dt
from numpy import (
    linspace, array,
    exp, log, sum, pi,
    matrix, zeros, arange
)
from numpy.linalg import inv as inverse
from numpy.linalg import norm
from numpy.random import seed as set_random_seed

from util import (
    add_noise, Objective,
    TimeSeries, nrow, cd,
    stringify)

"""
Want to estimate the parameters of a heating model given a
time series of temperature readings. Parameters will likely
include equilibrium temperature (T-infinity) and a rate constant
(h*A) / (m Cp) in the case of free convection.

Test1: Given some simulated data, can I estimate the parameters 
that I think are estimable?
"""


def temperature(t, a, b, c):
    """
    Parametric, time dependent temperature function
    :param t: The time(s) at which you want to know the temperature; the variable t
    :param a: the temperature far away (free stream temp)
    :param b: initial object temperature minus asymptotic temperature
        the constant that multiplies the exponential term
    :param c: positive real number, ratio of dT/dt to temperature
        difference between the object and its surroundings
        (this is the multiplicative constant in the governing
        differential equation).
    :return: temperature(s) without noise; theoretical temperatures
    """
    return a + b * exp(-c * t)


def nloglik(time_series, a, b, c, sigma):
    """
    negative log likelihood of observed time series
    under the normal errors model.
    :param time_series: TimeSeries object
    :param a: temperature far away;
        parameter to temperature function
    :param b: initial temperature (a t=0)
    :param c: rate constant for temperature function
    :param sigma: gaussian noise parameter (standard deviation)
    :return: float. negative log likelihood of parameters given
        observed time series
    """
    n = time_series.n
    times = time_series.times
    temps = time_series.temperatures
    sig_sq = sigma ** 2
    errors = temperature(times, a, b, c) - temps
    l = sum(errors ** 2) / (2 * sig_sq)
    l += n/2 * log(2 * pi * sig_sq)
    return l


def grad(time_series, a, b, c, t, sigma):
    """
    Compute the gradient of the negative log likelihood
    at some point (a,b,c,sigma)
    :param time_series: observed time series
    :param a: temperature far away
    :param b: initial temperature difference
    :param c: rate constant
    :param t: divisor for the log barrier on c
    :param sigma: noise parameter
    :return: gradient vector. numpy.ndarray
    """
    n = time_series.n
    times  = time_series.times
    temps  = time_series.temperatures
    errors = temperature(times, a, b, c) - temps
    sig_sq = sigma ** 2

    u  = exp(-c * times)
    v  = -times * b * u

    g = matrix(zeros((3, 1)))

    g[0, 0] = sum(errors) / sig_sq
    g[1, 0] = sum(errors * u) / sig_sq
    g[2, 0] = sum(errors * v) / sig_sq - (c * t) ** -1

    # This extra term (c*t)**-1 in the partial derivative wrt c
    # comes from the log barrier function to keep c positive.
    # Note that I have failed to include this term in the 2nd derivatives below.

    return g


def hessian(time_series, a, b, c, t, sigma):
    """
    A matrix of partial derivatives
    :param time_series: observed time series
    :param a: temperature far away
    :param b: initial temperature difference
    :param c: rate constant
    :param t: divisor for the log barrier on c
    :param sigma: noise parameter
    :return: a matrix of partial derivatives
    """
    n = time_series.n
    times = time_series.times
    temps = time_series.temperatures
    errors = temperature(times, a, b, c) - temps
    sig_sq = sigma ** 2

    u = exp(-c * times)
    v = -times * b * u

    h = matrix(zeros((3, 3)))

    h[0, 0] = n / sig_sq
    h[1, 1] = sum(u ** 2) / sig_sq
    h[2, 2] = sum(v ** 2 + errors * times ** 2 * b * u) / sig_sq

    h[0, 1] = h[1, 0] = sum(u) / sig_sq
    h[0, 2] = h[2, 0] = sum(v) / sig_sq
    h[1, 2] = h[2, 1] = sum(u * (v - errors * times)) / sig_sq

    return h


class Simulation(object):
    def __init__(self, t_init, t_hot, rate_const, sigma=0.):
        """
        Use this to produce a simulated time series of temperature data.

        Example:
        >>> s = Simulation(33, 475, .05, 3)
        >>> ts = s.simulate(2, 65)

        :param [degrees F] t_init: Initial temperature
        :param [degrees F] t_hot: Far away temperature (asymptotic temperature)
        :param [complicated units] rate_const: ratio of cooling rate to distance
            from asymptotic temperature (the rate constant in governing
            differential equation)
        :param [degrees F] sigma: standard deviation of temperature reading noise
        """
        # create the protected members
        self._t_init = None
        self._t_hot = None
        self._rate_const = None
        self._sigma = None
        self._noise_variance = None
        # assign values using property setters
        self.t_init = t_init
        self.t_hot = t_hot
        self.rate_const = rate_const
        self.sigma = sigma

    @staticmethod
    def times(t_total, n_pt):
        """
        Produce a list of times
        :param t_total: total elapse time
        :param n_pt: number of time points (number of periods + 1)
        :return: numpy.ndarray (1-d)
        """
        _times = linspace(
            start=0,
            stop=t_total,
            num=n_pt,
            dtype=dt
        )
        return _times

    def simulate(self, t_total, n_pt, random_seed=123):
        """
        Simulate the heating by convection
        :param t_total: total elapse time
        :param n_pt: number of time points, including the zero-time.
        :param random_seed: random seed (integer)
        :return: TimeSeries. An array of temperatures and times
        """
        times = self.times(t_total, n_pt)
        temps = temperature(
            t=times,
            a=self.t_hot,
            b=(self.t_init - self.t_hot),
            c=self.rate_const
        )
        set_random_seed(random_seed)
        add_noise(temps, self.sigma)
        return TimeSeries.from_time_temp(times, temps)

    def plot_time_series(self, t_total=None,
                         n_pt=None, random_seed=123,
                         time_series=None):
        """
        Plot a time series of temperatures
        :param t_total: total elapse time
        :param n_pt: number of time points, including the zero-time.
        :param random_seed: random seed (integer)
        :param time_series: Optional time series (numpy.ndarray)
            If present, other args are ignored
        :return: None
        """
        if time_series is None:
            time_series = self.simulate(
                t_total=t_total, n_pt=n_pt,
                random_seed=random_seed
            )
        time_series.plot(add_labels=True)
        plt.show()

    @property
    def t_init(self):
        return self._t_init

    @t_init.setter
    def t_init(self, value):
        if value < 0:
            msg = "Initial Temperature must be a positive " \
                  "number. This is absolute temperature"
            raise ValueError(msg)
        else:
            self._t_init = value

    @property
    def t_hot(self):
        return self._t_hot

    @t_hot.setter
    def t_hot(self, value):
        if value < 0:
            msg = "Hot temperature must be positive. " \
                  "This is absolute."
            raise ValueError(msg)
        else:
            self._t_hot = float(value)

    @property
    def rate_const(self):
        return self._rate_const

    @rate_const.setter
    def rate_const(self, value):
        if value < 0:
            msg = "Rate Constant must be positive. " \
                  "1st Law of Thermodynamics."
            raise ValueError(msg)
        else:
            self._rate_const = float(value)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        value = float(value)
        self._sigma = abs(value)
        self._noise_variance = value ** 2

    @property
    def noise_variance(self):
        """No setter for this one"""
        return self._noise_variance


class Optimization(object):
    def __init__(self, objective):
        """
        :param objective: the function whose value is to be minimized by adjusting parameters
        """
        assert isinstance(objective, Objective)
        self.objective = objective
        self.iterates = []  # for storing the point at each iteration
        self.iter_values = []  # for storing the function values
        self.iter_gradnorm = []  # for storing the norm of the gradient
        self.iter_hesscond = []  # for the condition number of the hessian matrix at each step
        self.optimal_point = None
        self.optimal_value = None
        # The variables that change at each iteration
        # keys are names of properties of this class
        self.iter_vars = {
            "values": "Objective Value",
            "hesscond": "Hessian Condition Number",
            "gradnorm": "Norm of Gradient"
        }

    def solve_newton(self, x0, t=1., tol=.000001, max_iter=500):
        """
        Estimate the parameters of the time dependent model given
            some observed time series. It is assumed that the first
            argument to the model is time while the remaining args
            are parameters
        :param x0: a starting point for the optimization
        :param t: newton step size multiplier
        :param tol: stopping criterion; tolerance on the norm of the gradient
        :param max_iter: maximum number of iterations to perform
        :return: Estimates of the parameters that would be input
            to the time dependent model
        """
        grad_norm = norm(self.objective.gradient(x0))
        x = array(x0, copy=True, dtype=dt)
        self.store_iteration(x)
        k = 0
        while grad_norm > tol:
            h = self.objective.hessian(x)
            hinv = inverse(h)
            g = self.objective.gradient(x)
            grad_norm = norm(g)
            direction = array(matrix(hinv) * matrix(g)).squeeze()
            x -= t * direction
            self.store_iteration(x)
            k += 1
            if k >= max_iter:
                break
        self.optimal_point = x
        self.optimal_value = self.objective.value(x)

    def store_iteration(self, x):
        """
        Store an iteration (point, function value, norm of gradient)
        :param x: the current point in the optimization
        :return: None
        """
        self.iterates.append(array(x, copy=True))
        self.iter_values.append(self.objective.value(x))
        self.iter_gradnorm.append(norm(self.objective.gradient(x)))
        self.iter_hesscond.append(self.objective.hessian_cn(x))

    @property
    def values(self):
        """
        Return the objective values at each iteration
        :return: numpy.ndarray
        """
        return array(self.iter_values)

    @property
    def hesscond(self):
        """
        Return the condition number of the hessian matrix at each iterate
        """
        return array(self.iter_hesscond)

    @property
    def gradnorm(self):
        """
        Return the L2 norm of the gradient at each iterate
        """
        return array(self.iter_gradnorm)

    @property
    def x0(self):
        """
        return the initial point
        :return: numpy.ndarray
        """
        return self.iterates[0]

    @property
    def iterations(self):
        """
        number of iterations that the optimization performed
        :return: int
        """
        return len(self.iterates) - 1

    @property
    def as_array(self):
        return array(self.iterates)

    def report_results(self):
        """
        Print the results of the optimization
        :return: None
        """
        print("Completed {:} optimization iterations".format(self.iterations))
        print("Optimal Point: ({:})".format(self.optimal_point))


class McPlotter(object):
    def __init__(self, optimization):
        """
        A class to plot the results of an optimization
        :param optimization: the optimization object
        """
        assert isinstance(optimization, Optimization)
        self.opt = optimization

    def summarize(self, run_name="optimization"):
        a, b, c = self.opt.x0
        params = {"a0": a,
                  "b0": b,
                  "c0": c}
        with cd(run_name):
            with cd("times_series_convergence"):
                fn1 = "timeseries_" + stringify(**params) + ".png"
                self.plot_time_series_convergence(file_name=fn1)
            with cd("parameter_convergence"):
                fn2 = "param_cnvg_" + stringify(**params) + ".png"
                self.plot_parameter_convergence(file_name=fn2)

    def plot_time_series_convergence(self, file_name=None, colorby=None):
        """
        Plot the observed time series along with the time series model
            at each step of the likelihood maximization
        :param file_name: file name of image to write
        :return: None
        """
        times = self.get_times()
        k = 1
        for params, color in ColorPicker(self.opt.iterates):
            temps = temperature(times, *params)
            TimeSeries.from_time_temp(times, temps).plot(
                _type="line",
                color=color,
                layer=k
            )
            k += 1
        self.plot_observed(layer=k)
        self.make_plot(file_name)

    def plot_parameter_convergence(self, file_name=None, colorby="hesscond"):
        """
        Plot the parameter estimates at each iterate
        :param file_name: str
        :param colorby: the variable to map to colors
        :return: None
        """
        points = self.opt.as_array
        n = nrow(points)
        if colorby in self.opt.iter_vars.keys():
            c = self.opt.__getattribute__(colorby)
        else:
            c = arange(n)
        scatter(
            points[:, 0],
            log(points[:, 2]),
            c=c, cmap='viridis',
            norm=colors.LogNorm(),
            edgecolors="black",
            alpha=1.0
        )
        plt.colorbar(label=self.opt.iter_vars[colorby])
        plt.xlabel('a')
        plt.ylabel('log(c)')
        plt.title('Parameter Convergence')
        self.make_plot(file_name)

    def plot_gradnorm_series(self):
        """
        Plot the norm of the gradient as a function of step number
        :return: None
        """
        values = self.opt.gradnorm
        scatter(
            x=range(len(values)),
            y=values
        )
        plt.title("Convergence Plot")
        plt.xlabel("Iteration Number")
        plt.ylabel("Norm of Gradient")
        plt.show()

    @staticmethod
    def make_plot(file_name):
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)
            plt.close()

    def get_times(self, n=100):
        _start, _stop = self.opt.objective.observed_data.range
        return linspace(_start, _stop, n)

    def plot_observed(self, layer=1):
        self.opt.objective.observed_data.plot(
            add_labels=True, layer=layer
        )


class ColorPicker(object):
    def __init__(self, iterates, cmap_name=None):
        """
        Iterate over objects
        returns tuple of object and color
        :param iterates: some iterable
        :param cmap_name: name of the color map
        """
        self.iterates = iterates
        self.cmap = get_cmap(cmap_name)
        self.norm = Normalize()

    @property
    def n(self):
        return self.iterates.__len__()

    def __iter__(self):
        for item, u in zip(self.iterates, linspace(0., 1., self.n)):
            color = self.cmap(u)
            yield item, color


if __name__ == "__main__":
    x0 = [800, -200, .003]  # Initial guess for optimization

    xt = [500, -400, .004]  # theoretical parameters (used in the simulation)
    rs = 100  # random seed
    sigma = 6
    newton_t = 0.25
    barrier_t = 1000
    sample_period = 20  # seconds
    samples = 65  # number of samples in the observed time series
    seconds = sample_period * samples

    # dictify for stringify
    other_params = dict(
        xt=xt,
        rs=rs,
        sg=sigma,
        tn=newton_t,
        tb=barrier_t,
        sp=sample_period,
        n=samples,
    )

    s = Simulation(
        t_init=xt[0] + xt[1],
        t_hot=xt[0],
        rate_const=xt[2],
        sigma=sigma
    )

    time_series = s.simulate(
        t_total=seconds,
        n_pt=samples,
        random_seed=rs
    )

    objective = Objective(
        func=nloglik,
        grad_f=grad,
        hess_f=hessian,
        observed_data=time_series,
        sigma=sigma,
        barrier_t=barrier_t
    )

    opt = Optimization(objective)
    opt.solve_newton(x0=x0, t=newton_t)
    opt.report_results()
    mcplot = McPlotter(opt)
    mcplot.summarize(run_name=stringify(**other_params))


