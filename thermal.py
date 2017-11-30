# THE NUMPY IMPORTS ARE RIGHT HERE!
import matplotlib.pyplot as plt
# PICTURE-MAKERS
from matplotlib.pyplot import scatter
from numpy import (
    linspace, array,
    exp, log, sum, ndarray,
    pi, matrix, zeros, squeeze
)
from numpy.linalg import inv as inverse
from numpy.linalg import norm
# RANDOMIZING FUNCTIONS FROM NUMPY.RANDOM
from numpy.random import seed as set_random_seed
from util import add_noise, column_bind, ncol, nrow

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


def grad(time_series, a, b, c, sigma):
    """
    Compute the gradient of the negative log likelihood
    at some point (a,b,c,sigma)
    :param time_series: observed time series
    :param a: temperature far away
    :param b: initial temperature difference
    :param c: rate constant
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
    g[2, 0] = sum(errors * v) / sig_sq
    return g


def hessian(time_series, a, b, c, sigma):
    """
    A matrix of partial derivatives
    :param time_series: observed time series
    :param a: temperature far away
    :param b: initial temperature difference
    :param c: rate constant
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

    h[0, 0] = n
    h[1, 1] = sum(u ** 2)
    h[2, 2] = sum(v ** 2 + errors * times ** 2 * b * u)

    h[0, 1] = h[1, 0] = sum(u)
    h[0, 2] = h[2, 0] = sum(v)
    h[1, 2] = h[2, 1] = sum(u * (v - errors * times))

    # h[0, 3] = h[3, 0] = -2 * sum(errors ** 2) / sigma ** 3
    # h[1, 3] = h[3, 1] = -2 * sum(dl_db * errors) / sigma ** 3
    # h[2, 3] = h[3, 2] = -2 * sum(errors * dl_dc)
    # h[3, 3] = 3 * sum(errors ** 2) / sigma ** 4 - n / sig_sq

    h /= sig_sq

    return h


class Objective(object):
    def __init__(self, func, grad_f, hess_f, observed_data, sigma):

        """
        A class to represent the objective function
        the function and its derivatives must take a TimeSeries object
        as its first argument
        :param func: the function itself
        :param grad_f: the gradient vector of the objective (a vector valued function)
        :param hess_f: the hessian matrix of the objective (a matrix valued function)
        :param observed_data: TimeSeries object
        """
        self._objective = func
        self._grad = grad_f
        self._hess = hess_f
        self._observed_data = None
        self._sigma = None
        self.observed_data = observed_data
        self.sigma = sigma

    @property
    def observed_data(self):
        return self._observed_data

    @observed_data.setter
    def observed_data(self, value):
        assert isinstance(value, TimeSeries)
        self._observed_data = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        assert isinstance(value, float)
        self._sigma = abs(value)

    def value(self, x):
        """
        Return the function value at a given point
        dimension of x must be compatible with the positional arguments in self.objective
        :param x: the point at which the function will be evaluated
        :return: float
        """
        return self._objective(
            time_series=self.observed_data,
            a=x[0],
            b=x[1],
            c=x[2],
            sigma=self.sigma
        )

    def gradient(self, x):
        return self._grad(
            time_series=self.observed_data,
            a=x[0],
            b=x[1],
            c=x[2],
            sigma=self.sigma
        )

    def hessian(self, x):
        return self._hess(
            time_series=self.observed_data,
            a=x[0],
            b=x[1],
            c=x[2],
            sigma=self.sigma
        )


class TimeSeries(object):
    def __init__(self):
        """
        Represent a time series
        """
        self._array = None

    def plot(self, add_labels=False):
        scatter(
            x=self.times,
            y=self.temperatures,
            alpha=.8
        )
        if add_labels:
            self.set_plot_labels()
        plt.show()

    @staticmethod
    def set_plot_labels():
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (F)")
        plt.title('Temperature Time Series')

    @property
    def n(self):
        return nrow(self._array)

    @classmethod
    def from_time_temp(cls, times, temps):
        assert len(times) == len(temps)
        _ts = cls()
        _ts.series = column_bind(times, temps)
        return _ts

    @property
    def series(self):
        return self._array

    @series.setter
    def series(self, value):
        assert isinstance(value, ndarray)
        assert ncol(value) == 2
        self._array = value

    @property
    def times(self):
        return self._array[:, 0]

    @property
    def temperatures(self):
        return self._array[:, 1]


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
            num=n_pt
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
        scatter(
            x=time_series.times,
            y=time_series.temperatures,
            alpha=.8
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (F)")
        plt.title('Temperature Time Series')
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
            self._t_init = float(value)

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


class Optimizer(object):
    def __init__(self, objective):
        """
        :param objective: the function whose value is to be minimized by adjusting parameters
        """
        assert isinstance(objective, Objective)
        self.objective = objective
        self._iterates = []
        self.optimal_point = None
        self.optimal_value = None

    def solve_newton(self, x0, t=1., tol=.000001):
        """
        Estimate the parameters of the time dependent model given
            some observed time series. It is assumed that the first
            argument to the model is time while the remaining args
            are parameters
        :param x0: a starting point for the optimization
        :param t: newton step size parameter
        :param tol: stopping criterion; tolerance on the norm of the gradient
        :return: Estimates of the parameters that would be input
            to the time dependent model
        """
        d = self.objective.gradient(x0)  # the gradient at the initial point
        x = array(x0, copy=True)
        self.store_iteration(x)
        k = 0
        max_iter = 2000
        while norm(d) > tol:
            h = self.objective.hessian(x)
            hinv = inverse(h)
            g = self.objective.gradient(x)
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
        Store an iteration (point and function value)
        :param x: the current point in the optimization
        :return: None
        """
        self._iterates.append(
            {
                'point': array(x, copy=True),
                'value': self.objective.value(x)
            }
        )

    @property
    def iterations(self):
        """
        number of iterations that the optimization performed
        :return: int
        """
        return len(self._iterates) - 1

    def report_results(self):
        print("Completed {:} iterations".format(self.iterations))
        print("Optimal Point: ({:})".format(self.optimal_point))


if __name__ == "__main__":
    s = Simulation(
        t_init=50,
        t_hot=300,
        rate_const=3.5*10**-3,
        sigma=1.5
    )
    ts = s.simulate(
        t_total=30*60,
        n_pt=50,
        random_seed=1729
    )
    objective = Objective(
        func=nloglik,
        grad_f=grad,
        hess_f=hessian,
        observed_data=ts,
        sigma=1.5
    )
    opt = Optimizer(objective)
    opt.solve_newton(x0=[500, -200, .003])
    opt.report_results()
