from .mixin import mixin
from .metadata_mixin import MetadataMixin


@mixin
class LayerMixin(MetadataMixin):
    def __str__(self):
        return f'{self.__class__} Layer'

    def __repr__(self):
        return str(self)
