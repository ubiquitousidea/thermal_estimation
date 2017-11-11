from numpy import (
    linspace, array,
    exp, concatenate
)
from numpy import array_repr as represent
from numpy.random import randn as noise
from numpy.random import seed as set_random_seed
from matplotlib.pyplot import scatter, Figure, Axes


"""
Want to estimate the parameters of a heating model given a
time series of temperature readings. Parameters will likely
include equilibrium temperature (T-infinity) and a rate constant
(h*A) / (m Cp) in the case of free convection.

Test1: Given some simulated data, can I estimate the parameters 
that I think are estimable?
"""


def temperature(time, temp_far,
                temp_init, rate_const):
    """
    Parametric, time dependent temperature function
    :param time: The time(s) at which you want to know the temperature; the variable t
    :param temp_far: the temperature far away (free stream temp)
    :param temp_init: the initial observed temperature of the object
    :param rate_const: positive real number, ratio of dT/dt to temperature
        difference between the object and its surroundings
        (this is the multiplicative constant in the governing
        differential equation).
    :return: temperature(s) without noise; theoretical temperatures
    """
    return temp_far - (temp_far - temp_init) * exp(-rate_const * time)


def add_noise(arr):
    """
    Add noise to the elements of an array
    MODIFIES ARGUMENT IN PLACE
    :param arr: a numpy.ndarray
    :return: NoneType. Modifies argument in place
    """
    arr += noise(*arr.shape)


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


class Simulation(object):
    def __init__(self, t_init, t_hot, rate_const, sigma=0.):
        """
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
        # >/\/oo\/\<
        # items to make the class easier to use
        self._last_args = None

    def simulate(self, t_max, freq, random_seed=123):
        """
        Simulate the heating by convection

        s = Simulation(...)
        r = s.simulate(6000, 10)  # 6000 seconds, 10 samples per second
        # r is a 60001 by 2 matrix.
        # Zeroth column is times.
        # Oneth (second) column is Temperatures.

        :param t_max: end time relative to start (seconds)
        :param freq: the rate at which samples are taken (samples / second)
        :param random_seed: random seed (integer)
        :return: numpy.ndarray. A matrix of temperatures and times
        """
        # TODO: Add argument storage so sets of methods can share most recently used arg list
        # self.store_args(
        #     {
        #         "simulation_args":
        #             {
        #                 "t_max": t_max,
        #                 "freq": freq,
        #                 "random_seed": random_seed
        #             }
        #     }
        # )

        times = linspace(
            start=0,
            stop=t_max,
            num=t_max * freq + 1
        )

        temps = temperature(
            time=times,
            temp_far=self.t_hot,
            temp_init=self.t_init,
            rate_const=self.rate_const
        )

        set_random_seed(random_seed)
        add_noise(temps)

        return column_bind(times, temps)

    def plot_time_series(self, t_max, freq, random_seed=123):
        """
        Plot a time series of temperatures
        :param t_max: end time relative to start (seconds)
        :param freq: the rate at which samples are taken (samples / second)
        :param random_seed: random seed (integer)
        :return: None
        """
        time_series = self.simulate(
            t_max=t_max,
            freq=freq,
            random_seed=random_seed
        )

        scatter(
            time_series[:, 0],
            time_series[:, 1]
        )

    @property
    def t_init(self):
        return self._t_init

    @t_init.setter
    def t_init(self, value):
        if value < 0:
            raise ValueError("Initial Temperature must be a positive number. This is absolute temperature")
        else:
            self._t_init = float(value)

    @property
    def t_hot(self):
        return self._t_hot

    @t_hot.setter
    def t_hot(self, value):
        if value < 0:
            raise ValueError("Hot temperature must be positive. This is absolute.")
        else:
            self._t_hot = float(value)

    @property
    def rate_const(self):
        return self._rate_const

    @rate_const.setter
    def rate_const(self, value):
        if value < 0:
            raise ValueError("Rate Constant must be positive. 1st Law of Thermodynamics.")
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


if __name__ == "__main__":
    s = Simulation(60, 415, .0005, 3)
    output = s.simulate(6000, 2)  # 6000 seconds, 10 samples per second
    print(
        represent(output, precision=3)
    )
    s.plot_time_series()
