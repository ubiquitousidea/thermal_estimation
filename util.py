from numpy import array, concatenate, ndarray
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