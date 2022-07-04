from .mixin import mixin

from ..utils.typesafety import type_safe

# To compare with attribute inherited from object
OBJ_ATTRS = dir(object)


@type_safe
def ATTR_FILTER(attr: str) -> bool:
    '''
    Filter attributes based on 2 criteria
        1. It must be defined in the class, not inherited from object
        2. It must not be a dunder attribute

    Parameters:
        attr: str
            The attribute to check for filtering

    Returns
        bool: True if `attr` passes all criteria, False otherwise
    '''
    in_obj = attr in OBJ_ATTRS
    is_dunder = attr.startswith('__') and attr.endswith('__')
    return not any(in_obj, is_dunder)


@mixin  # Prevents instantiation
class MetadataMixin:
    '''
    Provides `get_metadata` method to get a dictionary of attribute names
    and values

    Methods:
        `get_metada() -> dict`:
            Accumulates and returns attributes as a dictionary

    NOTE:
        Does not return callables
    '''
    @type_safe
    def get_metadata(self) -> dict:
        '''
        Returns a dictionary of {attribute name : value} pairs
        Attributes are filtered according to the following criteria
            1. It must be defined in the class, not inherited from object
            2. It must not be a dunder (magic) method/attribute

        Parameters:
            None

        Returns
            dict: attribute names as the keys, and the corresponding values.
        '''
        attrs = tuple(filter(ATTR_FILTER, dir(self)))
        return dict((attr, getattr(self, attr)) for attr in attrs)
