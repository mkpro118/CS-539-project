import sys
from typing import Callable, Type, Union
from .typesafety import type_safe

# exports isn't defined yet, so we have to do this manually
# __all__ = ['export', ]


@type_safe
def export(fn_cls: Union[Callable, Type]) -> Callable:
    '''
    Add class or function to the __all__ attribute of the module and to the
    __name_to_symbol_map__ dictionary which automatically created
    Parameters:
        fn_cls: Union[Callable, Type]
            The function or class to export

    Returns:
        Callable: The same function or class after adding it to __all__

    Raises:
        TypeError: if fn_cls is not of type Callable or Type

    Example Usage:

        import sys

        # With functions
        @export
        def func():
            pass

        print(sys.modules[__name__].__all__)
        # ['func']
        print(sys.modules[__name__].__name_to_symbol_map__)
        # {'func': <function func at 0x00000165CB5FBD30>}

        # With classes
        @export
        class cls():
            pass

        print(sys.modules[__name__].__all__)
        # ['func', 'cls']
        print(sys.modules[__name__].__name_to_symbol_map__)
        # {
        #   'func': <function func at 0x00000165CB5FBD30>,
        #   'cls': <class '__main__.cls'>
        # }
    '''
    mod = sys.modules[fn_cls.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn_cls.__name__)
    else:
        mod.__all__ = [fn_cls.__name__]

    if hasattr(mod, '__name_to_symbol_map__'):
        mod.__name_to_symbol_map__[fn_cls.__name__] = fn_cls
    else:
        mod.__name_to_symbol_map__ = {fn_cls.__name__: fn_cls}
    return fn_cls


sys.modules[__name__].__name_to_symbol_map__ = {'export': export}
