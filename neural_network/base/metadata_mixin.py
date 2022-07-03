from .mixin import mixin

from ..utils.typesafety import type_safe

OBJ_ATTRS = dir(object)


@type_safe
def ATTR_FILTER(attr: str) -> bool:
    in_obj = attr in OBJ_ATTRS
    is_dunder = attr.startswith('__') and attr.endswith('__')
    is_callable = callable(attr)
    is_metadata_fn = attr == 'get_metadata'
    return not any(in_obj, is_dunder, is_callable, is_metadata_fn)


@mixin
class MetadataMixin:
    @type_safe(skip=('self',))
    def get_metadata(self) -> dict:
        attrs = tuple(filter(ATTR_FILTER, dir(self)))
        return dict((attr, getattr(self, attr)) for attr in attrs)
