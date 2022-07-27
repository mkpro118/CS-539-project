import numpy as np
from typing import Union, Iterable
from numbers import Integral

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export

# Checks if kernel is within the input
in_bounds = lambda t, f, x: t + f <= x

# Computes the output size
out_size = lambda x, f, p, s: (x - f + sum(p)) // s + 1

# Extracts the matrix for convolution
extract = lambda x, t, l, f: x[t:t + f, l: l + f]


@type_safe
@not_none
def _get_stride(stride: Union[int, Iterable]) -> tuple:
    if isinstance(stride, int):
        return (stride,) * 2

    if isinstance(stride, Iterable):
        stride = tuple(iter(stride))
        if not all(map(lambda x: isinstance(x, (int, Integral, np.integer)), stride)):
            raise ValueError('stride iterable must contain integers')
        if len(stride) == 1:
            return (stride[0],) * 2
        if len(stride) == 2:
            return stride[:2]
        else:
            raise ValueError('Iterables passed must have a length of 1 or 2')


@type_safe
@not_none
def _get_padding(padding: Union[int, Iterable, str]) -> tuple:
    if isinstance(padding, int):
        return (padding,) * 4

    if isinstance(padding, Iterable):
        padding = tuple(iter(padding))
        if not all(map(lambda x: isinstance(x, (int, Integral, np.integer)), padding)):
            raise ValueError('padding iterable must contain integers')
        if len(padding) == 1:
            return (padding[0],) * 4
        if len(padding) == 2:
            return (padding[0],) * 2 + (padding[1],) * 2
        if len(padding) == 4:
            return padding[:4]
        else:
            raise ValueError('Iterables passed must have a length of 1, 2 or 4')

    raise ValueError("Padding can be either 'SAME' or 'VALID', or an int or list of integers")


@type_safe
@not_none
def _pad(input_: np.ndarray, padding: tuple):
    return np.pad(
        input_,
        (padding[:2], padding[2:]),
        mode='constant',
        constant_values=0
    )


@type_safe
@not_none
def _validate_params(X, kernel):
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError('Kernel matrix must be a square matrix')

    if kernel.shape[0] > (m := min(X.shape)):
        raise ValueError(
            'Kernel matrix must be smaller than the smallest dimension'
            f'given kernel has size={kernel.shape[0]}, and input\'s smallest'
            f'dimension is {m}'
        )


@type_safe
@not_none(nullable=('stride', 'padding'))
@export
def convolve(X: np.ndarray,
             kernel: np.ndarray, *,
             stride: Union[int, Iterable] = None,
             padding: Union[int, Iterable, str] = None) -> np.ndarray:
    '''
    Performs a convolution operation given the image matrix and kernel

    Parameters:
        X: numpy.ndarray of shape (m, n)
            The image matrix to perform convolution on
        kernel: numpy.ndarray of shape (f, f), where f < min(m, n) / 2
            The kernel matrix
        stride: Union[int, Iterable], keyword only, default = None
            The number of pixels to move per operation
            if arg is an integer, kernel moves right and down by the same amount
            if arg is an iterable of length 1, it is treated as an integer
            if arg is an iterable of length 2, then value at index 0 is the vertical stride
                and value at index 1 is the horizontal stride
            if not specified, will be taken as 1
        padding: Union[int, Iterable, str], keyword only, default = None
            The padding to add around the input matrix
            if arg is an integer, the same padding is added to each side
            if arg is an iterable of length 1, it is treated as an integer
            if arg is an iterable of length 2, then value at index 0 is vertical padding
                and value at index 1 is horizontal padding
            if arg is an iterable of length 4, then value at index
                0 - top padding
                1 - bottom padding
                2 - left padding
                3 - right padding
            if arg is a string, it can take two values
                'SAME' - padding will be added so that output dimensions match input dimensions
                'VALID' - no padding will be added
            if not specified, will be taken as 'VALID'

    Returns:
        numpy.ndarray: The matrix resulting from the convolution operation
    '''
    _validate_params(X, kernel)

    stride = stride = _get_stride(stride or 1)
    padding = padding or 0

    try:
        if padding.lower() == 'same':
            output_height, output_width = X.shape
            kernel_size = len(kernel)
            p1 = ((stride[0] - 1) * output_height - stride[0] + kernel_size) / 2
            p2 = ((stride[1] - 1) * output_width - stride[1] + kernel_size) / 2
            padding = (
                np.ceil(p1).astype(int),
                np.floor(p1).astype(int),
                np.ceil(p2).astype(int),
                np.floor(p2).astype(int),
            )
        elif padding.lower() == 'valid':
            padding = (0,) * 4
            raise AttributeError('fall through to except')
    except AttributeError:
        padding = _get_padding(padding)

        kernel_size = len(kernel)

        output_height = out_size(X.shape[0], kernel_size, padding[:2], stride[0])
        output_width = out_size(X.shape[1], kernel_size, padding[2:], stride[1])

    # empty matrix
    output = np.zeros((output_height, output_width))

    # semantics
    input_ = _pad(X, padding)

    input_height, input_width = input_.shape

    # Code for convolution
    top, left, out_x, out_y = 0, 0, 0, 0
    while in_bounds(top, kernel_size, input_height):
        output[out_y, out_x] = np.sum(extract(input_, top, left, kernel_size) * kernel)
        out_x += 1
        left += stride[1]
        if not in_bounds(left, kernel_size, input_width):
            left, out_x, out_y = 0, 0, out_y + 1
            top += stride[0]
    return output


@type_safe
@not_none(nullable=('stride', 'padding'))
@export
def convolve_transpose(X: np.ndarray,
                       kernel: np.ndarray, *,
                       stride: Union[int, Iterable] = None,
                       padding: Union[int, Iterable, str] = None) -> np.ndarray:
    '''
    Performs a transpose convolution operation given the image matrix and kernel

    Parameters:
        X: numpy.ndarray of shape (m, n)
            The image matrix to perform transpose convolution on
        kernel: numpy.ndarray of shape (f, f), where f < min(m, n) / 2
            The kernel matrix
        stride: Union[int, Iterable], keyword only, default = None
            The number of pixels to move per operation
            if arg is an integer, kernel moves right and down by the same amount
            if arg is an iterable of length 1, it is treated as an integer
            if arg is an iterable of length 2, then value at index 0 is the vertical stride
                and value at index 1 is the horizontal stride
            if not specified, will be taken as 1
        padding: Union[int, Iterable, str], keyword only, default = None
            The padding to add around the input matrix
            if arg is an integer, the same padding is added to each side
            if arg is an iterable of length 1, it is treated as an integer
            if arg is an iterable of length 2, then value at index 0 is vertical padding
                and value at index 1 is horizontal padding
            if arg is an iterable of length 4, then value at index
                0 - top padding
                1 - bottom padding
                2 - left padding
                3 - right padding
            if arg is a string, it can take two values
                'SAME' - padding will be added so that output dimensions match input dimensions
                'VALID' - no padding will be added
            if not specified, will be taken as 'VALID'

    Returns:
        numpy.ndarray: The matrix resulting from the transpose convolution operation
    '''
    _validate_params(X, kernel)

    stride = stride = _get_stride(stride or 1)
    padding = padding or 0

    try:
        if padding.lower() == 'same':
            kernel_size = len(kernel)
            p1 = (kernel_size - stride[0]) / 2
            p2 = (kernel_size - stride[1]) / 2
            padding = (
                np.ceil(p1).astype(int),
                np.floor(p1).astype(int),
                np.ceil(p2).astype(int),
                np.floor(p2).astype(int),
            )
        elif padding.lower() == 'valid':
            padding = (0,) * 4
            raise AttributeError('fall through to except')
    except AttributeError:
        padding = _get_padding(padding)

    # semantics
    input_ = X

    i = -1
    while (i := i + 1) < input_.shape[0] - 1:
        for _ in range(1, stride[0]):
            input_ = np.insert(input_, (i := i + 1), 0, axis=0)

    i = -1
    while (i := i + 1) < input_.shape[1] - 1:
        for _ in range(1, stride[1]):
            input_ = np.insert(input_, (i := i + 1), 0, axis=1)

    input_ = _pad(input_, padding)

    output = convolve(input_, kernel, stride=1, padding=padding)
    return output[
        padding[0]: output.shape[0] - padding[1],
        padding[2]: output.shape[1] - padding[3]
    ]
