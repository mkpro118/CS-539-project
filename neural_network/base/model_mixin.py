from .mixin import mixin
from .metadata_mixin import MetadataMixin
from .save_mixin import SaveMixin


@mixin
class ModelMixin(MetadataMixin, SaveMixin):
    pass
