import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
from numpy import float128 as dt
from numpy import (
    linspace, array,
    exp, log, sum, pi, matrix, zeros
)
from numpy.linalg import inv as inverse
from numpy.linalg import norm
from numpy.random import seed as set_random_seed

from util import add_noise, Objective, TimeSeries

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


class Optimizer(object):
    def __init__(self, objective):
        """
        :param objective: the function whose value is to be minimized by adjusting parameters
        """
        assert isinstance(objective, Objective)
        self.objective = objective
        self._iterates = []  # for storing the point at each iteration
        self._iter_values = []  # for storing the function values
        self._iter_gradnorm = []  # for storing the norm of the gradient
        self.optimal_point = None
        self.optimal_value = None

    def solve_newton(self, x0, t=1., tol=.000001, max_iter=100):
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
        d = self.objective.gradient(x0)  # the gradient at the initial point
        x = array(x0, copy=True, dtype=dt)
        self.store_iteration(x)
        k = 0
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
        Store an iteration (point, function value, norm of gradient)
        :param x: the current point in the optimization
        :return: None
        """
        self._iterates.append(array(x, copy=True))
        self._iter_values.append(self.objective.value(x))
        self._iter_gradnorm.append(norm(self.objective.gradient(x)))

    def plot_convergence(self):
        """
        Plot the norm of the gradient as a function of step number
        :return: None
        """
        values = array(self._iter_gradnorm)
        scatter(
            x=range(len(values)),
            y=values
        )
        plt.title("Convergence Plot")
        plt.xlabel("Iteration Number")
        plt.ylabel("Norm of Gradient")
        plt.show()

    @property
    def iterations(self):
        """
        number of iterations that the optimization performed
        :return: int
        """
        return len(self._iterates) - 1

    def report_results(self):
        """
        Print the results of the optimization
        :return: None
        """
        print("Completed {:} iterations".format(self.iterations))
        print("Optimal Point: ({:})".format(self.optimal_point))


if __name__ == "__main__":
    sigma = 1.8

    s = Simulation(
        t_init=50,
        t_hot=300,
        rate_const=.0035,
        sigma=sigma
    )
    for randomseed in range(20):

        sample_period = 6
        samples = 50
        seconds = sample_period * samples

        ts = s.simulate(
            t_total=seconds,
            n_pt=samples,
            random_seed=randomseed
        )
        objective = Objective(
            func=nloglik,
            grad_f=grad,
            hess_f=hessian,
            observed_data=ts,
            sigma=sigma,
            t=2000
        )
        opt = Optimizer(objective)
        opt.solve_newton(x0=[1000, -800, .0001], max_iter=200, t=.5)
        opt.report_results()
