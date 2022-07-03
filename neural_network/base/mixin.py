from functools import wraps
from typing import Callable
from ..utils.typesafety import type_safe


@type_safe
def mixin(cls: type) -> type:

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
