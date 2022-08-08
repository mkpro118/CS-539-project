'''Allow classes to be accesed from package level imports'''

from .batch_normalization import BatchNormalization, __name_to_symbol_map__ as bn_sym_map
from .convolutional import Convolutional, __name_to_symbol_map__ as conv_sym_map
from .dense import Dense, __name_to_symbol_map__ as dense_sym_map
from .flatten import Flatten, __name_to_symbol_map__ as flatten_sym_map

__name_to_symbol_map__ = {
    **bn_sym_map,
    **conv_sym_map,
    **dense_sym_map,
    **flatten_sym_map,
}
