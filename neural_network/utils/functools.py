from typing import Callable, Union
from functools import wraps


def unwrap(func: Callable, depth: Union[bool, int] = True):
    while depth:
        try:
            func = func.__wrapped__
            # If depth is of type int, we unwrap to that depth
            if isinstance(depth, int):
                depth -= 1
        except AttributeError:
            break
    return func


class MethodInvalidator:
    invalid_methods = set()

    @staticmethod
    def register(func: Callable):
        '''

        '''
        MethodInvalidator.invalid_methods.add(func.__qualname__)

    @staticmethod
    def validate(func: Callable):
        if func.__qualname__ in MethodInvalidator.invalid_methods:
            MethodInvalidator.invalid_methods.remove(func.__qualname__)

    @staticmethod
    def is_invocable(func: Callable) -> bool:
        return func.__qualname__ not in MethodInvalidator.invalid_methods

    @staticmethod
    def check_validity(func: Callable = None, *, invalid_logic: Callable = None) -> Callable:
        def decorator(func):
            @wraps(func)
            def inner(self, *args, **kwargs):
                if not MethodInvalidator.is_invocable(func):
                    return invalid_logic(self, *args, **kwargs)
                return func(self, *args, **kwargs)
            return inner

        if invalid_logic is None:
            invalid_logic = lambda *_, **__: None
            return decorator(func)
        else:
            return decorator
