# THE NUMPY IMPORTS ARE RIGHT HERE!
import matplotlib.pyplot as plt
# PICTURE-MAKERS
from matplotlib.pyplot import scatter
from numpy import (
    linspace, array,
    exp, log, sum, ndarray,
    pi, matrix, zeros
)
from numpy.linalg import inv as inverse
from numpy.linalg import norm
# RANDOMIZING FUNCTIONS FROM NUMPY.RANDOM
from numpy.random import randn as noise
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


def temperature(time, temp_far,
                delta_init, rate_const):
    """
    Parametric, time dependent temperature function
    :param time: The time(s) at which you want to know the temperature; the variable t
    :param temp_far: the temperature far away (free stream temp)
    :param delta_init: initial object temperature minus asymptotic temperature
        the constant that multiplies the exponential term
    :param rate_const: positive real number, ratio of dT/dt to temperature
        difference between the object and its surroundings
        (this is the multiplicative constant in the governing
        differential equation).
    :return: temperature(s) without noise; theoretical temperatures
    """
    if isinstance(time, ndarray):
        time_array = time
    elif isinstance(time, TimeSeries):
        time_array = time.times
    else:
        raise TypeError
    return temp_far + delta_init * exp(-rate_const * time_array)


def nloglik(time_series, t_f, d_i, k, sigma):
    """
    negative log likelihood of observed time series
    under the normal errors model.
    :param time_series: TimeSeries object
    :param t_f: temperature far away;
        parameter to temperature function
    :param d_i: initial temperature (a t=0)
    :param k: rate constant for temperature function
    :param sigma: gaussian noise parameter (standard deviation)
    :return: float. negative log likelihood of
        observed time series
    """
    # TODO: Show that this is convex in (t_f, t_i, k, sigma)
    n = time_series.n
    theoretical_temps = temperature(time_series, t_f, d_i, k)
    squared_errors = (time_series.temperatures - theoretical_temps) ** 2
    l = sum(squared_errors) / (2 * sigma ** 2)
    l += n/2 * log(2 * pi * sigma ** 2)
    return l


def grad(time_series, t_f, d_i, k, sigma):
    """
    Compute the gradient of the negative log likelihood
    at some point (t_f, d_i, k, sigma)
    :param time_series: observed time series
    :param t_f: temperature far away
    :param d_i: initial temperature difference
    :param k: rate constant
    :param sigma: noise parameter
    :return: gradient vector. numpy.ndarray
    """
    # variables: t_f = 1, d_i = 2, k = 3
    times  = time_series.times
    temps  = time_series.temperatures
    errors = temperature(times, t_f, d_i, k) - temps
    sig_sq = sigma ** 2
    exps   = exp(-k * times)     # d error / d d_i
    v      = times * d_i * exps  # -d error / d k
    g = matrix(zeros((3, 1)))
    g[0, 0] = sum(errors) / sig_sq
    g[1, 0] = sum(errors * exps) / sig_sq
    g[2, 0] = -sum(errors * v) / sig_sq
    return g


def hessian(time_series, t_f, d_i, k, sigma):
    """
    A matrix of partial derivatives
    :param time_series: observed time series
    :param t_f: temperature far away
    :param d_i: initial temperature difference
    :param k: rate constant
    :param sigma: noise parameter
    :return: a matrix of partial derivatives
    """
    # variables: t_f = 1, d_i = 2, k = 3
    n = time_series.n
    times = time_series.times
    temps = time_series.temperatures
    errors = temperature(times, t_f, d_i, k) - temps
    sig_sq = sigma ** 2
    exps = exp(-k * times)  # d error / d d_i
    v = times * d_i * exps  # -d error / d k
    h = matrix(zeros((3, 3)))

    h[0, 0] = n / sig_sq
    h[1, 1] = sum(exps ** 2) / sig_sq
    h[2, 2] = sum(v * (v + errors * times)) / sig_sq

    h[0, 1] = h[1, 0] = sum(exps) / sig_sq
    h[0, 2] = h[2, 0] = -sum(v) / sig_sq
    h[1, 2] = h[2, 1] = -sum(exps * (v + errors * times))

    return h


class Objective(object):
    def __init__(self, _f, _grad_f, _hess_f, observed_data):

        """
        A class to represent the objective function
        the function and its derivatives must take a TimeSeries object
        as its first argument
        :param _f: the function itself
        :param _grad_f: the gradient vector of the objective
        :param _hess_f: the hessian matrix of the objective
        :param observed_data: TimeSeries object
        """
        self._objective = _f
        self._grad = _grad_f
        self._hess = _hess_f
        self._observed_data = None
        self.observed_data = observed_data

    @property
    def observed_data(self):
        return self._observed_data

    @observed_data.setter
    def observed_data(self, value):
        assert isinstance(value, TimeSeries)
        self._observed_data = value

    def value(self, x):
        """
        Return the function value at a given point
        dimension of x must be compatible with the positional arguments in self.objective
        :param x: the point at which the function will be evaluated
        :return: float
        """
        return self._objective(self.observed_data, *x)

    def gradient(self, x):
        return self._grad(self.observed_data, *x)

    def hessian(self, x):
        return self._hess(self.observed_data, *x)


class TimeSeries(object):
    def __init__(self):
        """
        Represent a time series
        """
        self._array = None

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
            time=times,
            temp_far=self.t_hot,
            delta_init=(self.t_init - self.t_hot),
            rate_const=self.rate_const
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
        self.solution = None

    def solve(self, x0, t=1., tol=.000001):
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
        iterates = [x0]
        while norm(d) > tol:
            h = self.objective.hessian(x)
            hinv = inverse(h)
            g = self.objective.gradient(x)
            direction = matrix(hinv) * matrix(g)
            x -= t * direction
            iterates.append(x)
        self.solution = x
        return iterates


if __name__ == "__main__":
    s = Simulation(
        t_init=70,
        t_hot=300,
        rate_const=3.5*10**-3,
        sigma=1.5
    )
    ts = s.simulate(
        t_total=30*60,
        n_pt=50,
        random_seed=144
    )
    s.plot_time_series(
        time_series=ts
    )
    objective = Objective(
        _f=nloglik,
        _grad_f=grad,
        _hess_f=hessian,
        observed_data=ts
    )
    # Investigate the shape of this function
    # Plot contours of constant value (2D slices)
    opt = Optimizer(objective)
    x0 = array([200, 100, .01, 1])  # Initial guess
    iterates = opt.solve(x0)












