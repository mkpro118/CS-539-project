from .mixin import mixin
from .metadata_mixin import MetadataMixin
from .save_mixin import SaveMixin


@mixin  # Prevents instantiation
class ModelMixin(MetadataMixin, SaveMixin):
    '''
    Mixin for easier definition of Model classes

    Inherited from MetadataMixin
        `get_metadata`

    Inherited from SaveMixin
        `save`
    '''
    pass
