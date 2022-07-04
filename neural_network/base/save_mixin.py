from typing import Any
from json import dumps

from .mixin import mixin
from ..utils.typesafety import type_safe


@mixin  # Prevents instantiation
class SaveMixin:
    '''
    Mixin to provide functionality to save data associated to the instance

    Methods:
        `save(filename: str, data: Any) -> None`:
    '''
    @type_safe
    def save(self, filename: str, data: Any = None) -> None:
        '''
        Save attributes of an instance into a file in json format

        Parameters:
            filename: str
                The name or path of the file to save to
            data: Any, default = None
                The data to save, if None, `get_metadata` from
                base.MetadataMixin will be tried. If that too is not found
                nothing will be saved

        See:
            `base.metadata_mixin.MetadataMixin`
        '''
        if data is None:
            # User didn't specify data
            try:
                # Trying to compute metadata using Metadata Mixin
                data = self.get_metadata()
            except AttributeError:
                # No dice with the MetadataMixin
                print(
                    f'There\'s no data to save! '
                    f'Inherit base.SaveMixin to automatically compute data '
                    f'or pass in data as the second positional argument'
                )
                return

        # Convert data to json format
        data_str = dumps(data, indent=4)

        # Write to file
        with open(filename, 'w') as f:
            f.write(data_str)
