from .mixin import mixin
from .metadata_mixin import MetadataMixin


@mixin  # Prevents instantiation
class LayerMixin(MetadataMixin):
    '''
    Mixin class for all Layers.

    Provides __repr__ and __str__ for the layers.

    Inherited from MetadataMixin
        method `get_metadata` to computer layer's metadata
    '''

    def __str__(self):
        return f'{self.__class__} Layer'

    def __repr__(self):
        return str(self)
