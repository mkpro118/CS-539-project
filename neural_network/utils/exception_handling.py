from functools import wraps
from typing import Callable, Union
from traceback import format_exception
try:
    from typesafety import type_safe
except ImportError:
    from .typesafety import type_safe

# default message if a message dict is missing a certain exception
DEFAULT_MSG = lambda e: f'''No Error message specified for {str(type(e)).split("'")[1]}'''

# format exception traceback
TRACEBACK_FORMATTER = lambda e: format_exception(etype=type(e), value=e, tb=e.__traceback__)


@type_safe
def safeguard(*exceptions,
              msg: Union[str, dict] = None,
              print_trace: bool = False) -> Callable:
    '''
    Decorator to handler exceptions for a Callable.

    Parameters:
        exceptions: tuple of Exception and it's subclasses
            Tuple of exceptions to handle
        msg: Union[str, dict]
            Error messages to display in case of an exception
            If msg is of type str
                The same message is shown for all exceptions
            If msg is of type dict
                The message corresponding to the exception is shown

    Returns:
        Callable: Decorated function to handle given exceptions

    Raises:
        ValueError: if no exceptions are specified
        TypeError: if a parameter is not a subclass of Exception

    Example Usage:

        @safeguard(ZeroDivisionError, msg='Cannot divide by 0')
        def divide(x: float, y: float) -> float:
            return x / y

        divide(10.0, 5.0) # returns 2.0
        divide(1, 0) # returns None, prints 'Cannot divide by 0'
    '''
    # check exceptions is not empty
    if not len(exceptions) > 0:
        raise ValueError('Need at least 1 exception to safeguard against!')

    # check all exceptions are actually exceptions
    for exception in exceptions:
        if not issubclass(exception, Exception):
            raise TypeError(f'{exception} is not a subclass of {Exception}')

    def decorator(func: Callable):
        '''
        Actual decorator
        '''
        @ wraps(func)
        def wrapper(*args, **kwargs):
            '''
            Wrapper that contains the exception handling logic
            '''

            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if print_trace:
                    print(''.join(TRACEBACK_FORMATTER(e)))
                else:
                    # print a smaller error message
                    exc_type = str(type(e)).split("'")[1]
                    print(f'(Exception in <{func.__qualname__}>) {exc_type}: {e}')
                if msg:
                    if isinstance(msg, str):
                        print(f'Error message: "{msg}"')
                    else:
                        print(f'Error message: "{msg.get(type(e), DEFAULT_MSG)}"')
        return wrapper
    return decorator


@type_safe
def warn(msg: str) -> Callable:
    '''
    Decorator to display a warning before executing a function

    Parameters:
        msg: str
            The warning message to display

    Returns:
        Callable: A decorated Callable which displays a warning before execution

    Example Usage

        @warn(msg='ZeroDivisionError possible')
        def divide(x: float, y: float) -> float:
            return x / y

        divide(10.0, 5.0) # displays 'ZeroDivisionError possible' before 2.0
    '''
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # print the warning
            print(f'Warning {func.__qualname__}: {msg}')
            return func(*args, **kwargs)
        return wrapper
    return decorator
