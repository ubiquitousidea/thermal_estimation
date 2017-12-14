import os
import json
from matplotlib import pyplot as plt
from matplotlib.pyplot import scatter
from numpy import array, concatenate, ndarray, zeros, nan
from numpy.linalg import svd
from numpy.linalg.linalg import LinAlgError
from numpy.random.mtrand import randn as noise
from contextlib import contextmanager


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


@contextmanager
def cd(new_directory=None):
    """
    From Stack Exchange example
    https://stackoverflow.com/questions/431684/how-do-i-cd-in-python
    :param new_directory: Directory to change into
    """
    if new_directory is None:
        new_directory = "."
    previous_directory = os.getcwd()
    if not os.path.isdir(new_directory):
        os.mkdir(new_directory)
    new_directory = os.path.expanduser(new_directory)
    os.chdir(new_directory)
    try:
        yield
    finally:
        os.chdir(previous_directory)


class Objective(object):
    def __init__(self, func, grad_f, hess_f,
                 observed_data, sigma):

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
            sigma=self.sigma
        )
        return h

    def hessian_cn(self, x):
        """
        Condition number of the hessian matrix
        :return: float (NaN if hessian is not invertible)
        """
        h = self.hessian(x)
        try:
            u, s, vt = svd(h)
        except LinAlgError:
            return nan
        s = relu(s)
        if s.min() > 0:
            return s.max() / s.min()
        else:
            return nan

    def contour_plot(self, plot=False, fname=None):
        """
        Produce a contour plot of the objective function on the current figure
        :param plot: If true, call pyplot.show()
        :param fname: If name, write a png image of the contour plot
        """
        # TODO: Add the contour plot functionality to the objective class

    @property
    def observed_data(self):
        return self._observed_data

    @observed_data.setter
    def observed_data(self, value):
        if value is None:
            self._observed_data = None
        elif isinstance(value, TimeSeries):
            self._observed_data = value
        else:
            raise TypeError(
                "time series value must be either "
                "TimeSeries instance or None"
            )

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

    def plot(self, add_labels=False,
             _type="scatter", color=None,
             layer=1, edgecolor='black'):
        """
        Plot the time series
        :param add_labels: If True, axis labels and a title will be added.
        :param _type: plot type; one of (scatter, line)
        :param color: the color for the plotted points/lines (optional)
        :param layer: which layer to plot the objects (higher number displays on top)
        :return: None
        """
        _type = _type.lower()
        assert _type in ("scatter", "line")
        if _type == "scatter":
            scatter(
                self.times,
                self.temperatures,
                alpha=.8,
                zorder=layer,
                edgecolors=edgecolor
            )
        elif _type == "line":
            plt.plot(self.times,
                     self.temperatures,
                     "-", color=color,
                     zorder=layer)
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


def stringify(**kwargs):
    """
    Convert a namespace into a nice readable string
    :param kwargs: arbitrary keyword arguments
    :return: string representation of the names:values
    """
    return "_".join(
        [
            "{}-{}".format(k, kwargs[k])
            for k
            in sorted(kwargs.keys())
        ]
    )


def relu(arr):
    """
    return the max of {0, xi} for each i in range of array
    :param arr: numpy.ndarray
    :return: numpy.ndarray with all non-negative entries
    """
    return column_bind(zeros(len(arr)), arr).max(axis=1)


def to_json(arr_like, fname):
    """
    Write out the contents of the arraylike object to the file fname
    :param arr_like: array like object
    :param fname: file name
    """
    arr = array(arr_like).tolist()
    with open(fname, 'w')as fh:
        fh.write(json.dumps(arr))


def from_json(fname):
    """
    read a json file and return a dict
    :param fname: file name
    :return: dict
    """
    with open(fname, 'r') as fh:
        d = json.load(fh)
    return d
