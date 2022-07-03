from typing import Any

from .mixin import mixin
from ..utils.typesafety import type_safe


@mixin
class SaveMixin:

    @type_safe
    def save(self, filename: str, data: Any):
        with open('filename', 'w') as f:
            f.write(data)
