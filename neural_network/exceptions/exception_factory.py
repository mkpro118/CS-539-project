from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from ..base.mixin import mixin


@mixin
@export
class ExceptionFactory:
    '''
    Internal factory for creating exceptions
    '''

    @classmethod
    @type_safe
    @not_none
    def register(cls, name: str, **kwargs) -> type:
        '''
        Creates a new exception, or returns an already created exception

        Parameters:
            name: str
                The name of the exception to create
            kwargs: dict
                Additional arguments for creating the exception

        Returns:
            type: An exception class of the given name
        '''
        try:
            return getattr(cls, name)
        except AttributeError:
            kwargs.update({
                '__reduce__': lambda self: (
                    _NestedClassHelper(),
                    (ExceptionFactory, self.__class__.__name__, ),
                    self.__dict__.copy(),
                )
            })
            exception = type(name, (Exception,), kwargs)
            setattr(cls, name, exception)
            return exception


class _NestedClassHelper(Exception):
    '''
    To allow access of nested classes from ExceptionFactory
    for pickling by ProcessPoolExecutor
    '''

    @type_safe
    @not_none
    def __call__(self, class_: type, name: str) -> '_NestedClassHelper':
        '''
        Used by __reduce__ in pickling

        Parameters:
            class_: type
                The parent class of the nested class
            name: str
                The name of the nested class

        Returns:
            _NestedClassHelper: instance with it's class assignment as the nested class
        '''
        class_, instance = getattr(class_, name), _NestedClassHelper()
        instance.__class__ = class_
        return instance
