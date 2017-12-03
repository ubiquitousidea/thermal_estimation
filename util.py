from matplotlib import pyplot as plt
from matplotlib.pyplot import scatter
from numpy import array, concatenate, ndarray, inf
from numpy.random.mtrand import randn as noise


def add_noise(arr, sigma):
    """
    Add noise to the elements of an array
    MODIFIES ARGUMENT IN PLACE
    :param arr: a numpy.ndarray
    :return: NoneType. Modifies argument in place
    """
    dims = arr.shape
    arr += sigma * noise(*dims)


def column_bind(arr1, arr2):
    """
    Take two one-dimensional arrays,
        glue them together, and return a nx2 array

    Name chosen to be consistent with R

    :param arr1: numpy.ndarray
    :param arr2: numpy.ndarray
    :return: numpy.ndarray
    """
    arr1 = array(arr1, ndmin=2)
    arr2 = array(arr2, ndmin=2)
    _out = concatenate((arr1, arr2), axis=0).transpose()
    return _out


def ncol(arr):
    """
    Nice R function
    Made to raise errors when input is not a 2d array
    :param arr: array
    :return: number of columns in array
    """
    assert isinstance(arr, ndarray)
    assert len(arr.shape) == 2
    return arr.shape[1]


def nrow(arr):
    """
    Another nice, readable R function
    Made to raise errors when input is not a 2d array
    :param arr: array
    :return: number of rows in array
    """
    assert isinstance(arr, ndarray)
    assert len(arr.shape) == 2
    return arr.shape[0]

#
# class Constraint(object):
#     def __init__(self, typ="=", ):
#

class Objective(object):
    def __init__(self, func, grad_f, hess_f, observed_data,
                 sigma, barrier_t=inf):

        """
        A class to represent the objective function
        the function and its derivatives must take a
        TimeSeries object as their first argument.
        :param func: the function itself
        :param grad_f: the gradient vector of the objective (a vector valued function)
        :param hess_f: the hessian matrix of the objective (a matrix valued function)
        :param observed_data: TimeSeries object
        :param sigma: the noise parameter for the function (assumed to be known)
        :param barrier_t: divisor of the log barrier function. Increase this to
            steepen the penalty for violating a constraint
        """
        self._objective = func
        self._grad = grad_f
        self._hess = hess_f
        self._t = barrier_t
        self._observed_data = None
        self._sigma = None
        self.observed_data = observed_data
        self.sigma = sigma

    def value(self, x):
        """
        Return the function value at a given point
        :param x: the point at which the function will be evaluated
        :return: float
        """
        f = self._objective(
            time_series=self.observed_data,
            a=x[0],
            b=x[1],
            c=x[2],
            sigma=self.sigma
        )
        return f

    def gradient(self, x):
        """
        Compute the gradient of the objective at the point x
        :param x: the point at which the gradient vector will be evaluated.
        :return: numpy.ndarray; shape=(3,1)
        """
        g = self._grad(
            time_series=self.observed_data,
            a=x[0],
            b=x[1],
            c=x[2],
            t=self._t,
            sigma=self.sigma
        )
        return g

    def hessian(self, x):
        """
        Compute the matrix of 2nd partial derivatives at x
        :param x: the point at which the Hessian matrix will be computed
        :return: numpy.ndarray; shape=(3,1)
        """
        h = self._hess(
            time_series=self.observed_data,
            a=x[0],
            b=x[1],
            c=x[2],
            t=self._t,
            sigma=self.sigma
        )
        return h

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
        self._sigma = abs(float(value))


class TimeSeries(object):
    def __init__(self):
        """
        Represent a time series
        """
        self._array = None

    def plot(self, add_labels=False, _type="scatter", color=None):
        """
        Plot the time series
        :param add_labels: If True, axis labels and a title will be added.
        :return: None
        """
        _type = _type.lower()
        assert _type in ("scatter", "line")
        if _type == "scatter":
            scatter(
                self.times,
                self.temperatures,
            )
        elif _type == "line":
            plt.plot(self.times,
                     self.temperatures,
                     "-", color=color)
        if add_labels:
            self.set_plot_labels()

    @property
    def range(self):
        """
        Return the range of times represented by this timeseries
        :return: tuple of floats
        """
        return self.times[0], self.times[-1]

    @staticmethod
    def set_plot_labels():
        """
        Add labels to the current figure
        :return: None
        """
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (F)")
        plt.title('Temperature Time Series')

    @property
    def n(self):
        """
        Return the number of time points in the time series
        :return: int
        """
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