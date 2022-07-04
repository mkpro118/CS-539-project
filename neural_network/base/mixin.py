from functools import wraps
from typing import Callable
from ..utils.typesafety import type_safe


@type_safe
def mixin(cls: type) -> type:
    '''
    Marks a class as a mixin and prevents the instantiation of the mixin class

    Doesn't affect the existing `__init__` method defined by the Mixin class,
    just decorates it to check that the object is not of the type defined by
    the mixin.
    '''

    @type_safe
    def mixin_decorator(func: Callable) -> Callable:

        @wraps(func)
        def mixin_wrapper(self, *args, **kwargs):
            if self.__class__ == cls:
                raise TypeError('Cannot instantiate a Mixin!')
            return func(self, *args, **kwargs)
        return mixin_wrapper

    cls.__init__ = mixin_decorator(cls.__init__)
    return cls
