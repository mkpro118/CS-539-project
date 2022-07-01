from functools import wraps
from inspect import getfullargspec
from typing import Callable, Iterable, Union, get_args as get_subscripts


# Add support for compatiblility with NoneType
NONE_TYPE = type(None)


def type_safe(func: Callable):
    '''
    This decorator ensures type safety over type annotated functions

    Ensures there's no type mismatch over the parameters which are
    annotated with types.

    Does not affect parameters that are not annotated, they're defaulted
    to be of type object.

    VarArgs are not type checked

    NoneType is considered to be a valid argument, use not_none
    along with this decorator to ensure null safety.

    Return types are also checked for type mismatch.

    Parameters:
        `func`: Callable
            The function to ensure type safety over

    Returns:
        Callable: a wrapped function with type safety

    Raises:
        TypeError: if there's a type mismatch

    Example Usage:
        @type_safe
        def integer_sum(x: int, y: int) -> int:
            return x + y

        integer_sum(1, 2) # returns 3
        integer_sum(1.5, 2.0) # raises a TypeError
        integer_sum('hello', False) # raises a TypeError


        # Since return types are also checked

        @type_safe
        def integer_divison(x: int, y: int) -> int:
            return x / y

        integer_sum(10, 2) # raises a TypeError


    Notes:
        * NoneTypes are NOT considered for type mismatch
            Consider using the not_none decorator for null safety
        * VarArgs are not type checked
        * Known Bug: Doesn't work on nested subscripted generic aliases
    '''

    # Ensure `func` is a callable to be decorated
    if not callable(func):
        raise TypeError(
            f'Parameter to type_safe() must be callable '
            f'(got "{func}" [{type(func)}])'
        )

    # Argument inspection
    specs = getfullargspec(func)

    # Function to get items from the annotation dictionary
    # `annotations` is actually the callable `get` method of a dict object
    annotations = specs.annotations.get

    # Keyword argument filter, type safety over that is not guaranteed
    kw_filter = lambda y: lambda x: x not in y

    def validate_types(*args: tuple[tuple]):
        '''
        Validates the types from the tuples in args
        '''
        nonlocal annotations

        # list of type errors, to show all errors at once
        errors = []

        # iterating over args and validating each component
        for req, real in args:
            try:
                try:
                    # get a tuple of subscripted annotations, may be empty
                    annotation = get_subscripts(annotations(req, object))

                    # assert that the tuple of subscripts is not empty
                    assert (n := len(annotation)) > 0
                except AssertionError:
                    # if it is empty, create a tuple of size 1
                    # in order to use the starred (*) expression
                    annotation = (annotations(req, object),)

                # assert there's no type mismatch
                assert isinstance(real, (*annotation, NONE_TYPE))
            except AssertionError:
                # store mismatches if any in the `errors` list
                q = "'"
                f_map = lambda x: f'{q}{x.split(q)[1]}{q}' if q in x else x
                n = len(annotation)
                errors.append((
                    f"Valid type{'s' if n > 1 else ''} for '{req}'",
                    f'{"are" if n > 1 else "is"} '
                    f'{", ".join(map(f_map, map(str, annotation)))}',
                    f", got {f_map(str(type(real)))} instead!",
                ))

        if len(errors) > 0:
            # error Message formatting
            max_l = [max(map(lambda x: len(x[i]), errors)) for i in range(3)]
            r = lambda x, i: x + ' ' * (max_l[i] - len(x))
            errors = map(lambda x: (r(x[0], 0), r(x[1], 1), r(x[2], 2)), errors)
            errors = '\n'.join(map(lambda x: ' '.join(x), errors))

            # raising a TypeError with the formatted error message
            raise TypeError(
                f'Function <{func.__qualname__}> invoked with incorrect types\n'
                f'{"-" * (sum(max_l) + 2)}\n'
                f"{errors}\n"
                f'{"-" * (sum(max_l) + 2)}\n'
            )

    @wraps(func)
    def wrapper(*args, **kwargs):
        '''
        Wrapper Function over `func` to provide type safety checks
        '''
        # Validate the positional arguments
        validate_types(*zip(filter(kw_filter(kwargs), specs.args), args))

        # Validate the keyword argument
        validate_types(*kwargs.items())

        # Calculate the result of func
        result = func(*args, **kwargs)

        # Validate the return type
        validate_types(('return', result, ))

        # Return the result
        return result

    # Return wrapper function
    return wrapper


# Sentinel object for function not_null's default parameter nullable
SENTINEL_TUPLE = tuple()


@type_safe
def not_none(nullable: Union[Callable, Iterable] = SENTINEL_TUPLE,
             *, follow_wrapped: Union[bool, int] = True) -> Callable:
    '''
    Decorator to avoid None values in function arguments
    Unwraps the function to the maximum possible depth by default

    Parameters:
        nullable: Union[Callable, Iterable]
            If used with parentheses:
                Parameters that can be Non null
            If used without parentheses (as a decorator):
                The function to decorate
        follow_wrapped: Union[bool, int]
            * Only available when not_none is called with parentheses
            If follow_wrapped is of type bool:
                Unwraps the function to the maximum possible depth
            If follow_wrapped is of type int:
                Unwraps the function to the follow_wrapped depth

    Returns:
        Callable: Decorated or Decorator function.
            If nullable is a Callable:
                A decorated function which checks
                that all parameter are not None
            If nullable is a Iterable:
                A decorator function, which can checks
                that all paramters except
                the ones in nullable are not None

    Raises:
        TypeError: if a non nullable parameter is None

    Example Usage:

        @not_none
        def add(x: int, y: int) -> int:
            return x + y

        add(1, 2) # returns 3
        add(1, None) # raises a TypeError
        add(None, 2) # raises a TypeError
        add(None, None) # raises a TypeError

        # nullable paramters can be specified as shown
        @not_none(nullable=["c", "d",])
        def func(a: int, b: str, c: int, d:str = None) -> None:
            pass

        func(1, 'Hi!', None) # works as expected
        func(1, 'Hi!', None, None) # works as expected
        func(1, None, None) # raises a TypeError
    '''
    @type_safe
    def decorator(func: Callable):
        '''
        Actual decorator for not None checks
        '''
        nonlocal nullables, follow_wrapped

        # Unwrap func in order to avoid messing with other decorators
        orig_func = func
        while follow_wrapped:
            try:
                orig_func = orig_func.__wrapped__
                # If follow_wrapped is of type int, we unwrap to that depth
                if isinstance(follow_wrapped, int):
                    follow_wrapped -= 1
            except AttributeError:
                break

        specs = getfullargspec(orig_func)
        kw_filter = lambda y: lambda x: x not in y

        def validate_args(*args: tuple[tuple]):
            '''
            Validates the params are not None from the tuples in args
            '''
            nonlocal nullables

            # List of None parameters errors, to show all errors at once
            errors = []

            # Iterating over args and validating each component
            for req, real in args:
                if req in nullables:
                    continue

                try:
                    assert real is not None
                except AssertionError:
                    errors.append(f"Parameter '{req}' cannot be {real}")

            if len(errors) > 0:
                # Error Message formatting
                errors = '\n'.join(errors)

                # Raising a TypeError with the formatted error message
                raise TypeError(
                    f'Function <{func.__qualname__}> invoked with '
                    f'{real} values on non nullable parameters\n'
                    f'{"-" * len(errors)}\n'
                    f"{errors}\n"
                    f'{"-" * len(errors)}\n'
                )

        @wraps(func)
        def wrapper(*args, **kwargs):
            '''
            Wrapper Function over `func` to provide not None checks
            '''
            nonlocal specs, kw_filter, validate_args, func

            # Validate the positional arguments
            validate_args(*zip(filter(kw_filter(kwargs), specs.args), args))

            # Return the result of func
            return func(*args, **kwargs)
        return wrapper

    if callable(nullable):
        nullables = tuple()
        return decorator(nullable)
    else:
        nullables = tuple(nullable)
        return decorator
