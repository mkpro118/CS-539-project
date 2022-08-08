'''
Utility function to load the dataset
    load_data(mode: str, *, shape: tuple, grayscale: bool, depth_first: bool) -> Data
'''

from typing import Union


def load_data(mode: str = 'original', *, return_X_y: bool = False,
              size: tuple = None, depth_first: bool = True) -> Union[dict, tuple]:
    '''
    Loads and returns the chess pieces dataset as a Data object (a dictionary-like object)
    The data can be customized to have any size, be colored or grayscaled


    =============================== ==============
    Classes                                      5
    Samples per class                     variable
      Class 0 (bishop)                         141
      Class 1 (knight)                         174
      Class 2 (pawn)                            82
      Class 3 (queen)                          115
      Class 4 (rook)                           139
    Samples total                              651
    Dimensionality                        variable
      If depth_first=True
        If mode='original'           (3, 224, 224)
        If mode='resized'                (3, h, w)
        If mode='grayscaled'         (1, 224, 224)
        If mode='resize grayscaled'      (1, h, w)
      If depth_first=False
        If mode='original'           (224, 224, 3)
        If mode='resized'              (custom, 3)
        If mode='grayscaled'         (224, 224, 1)
        If mode='resize grayscaled'      (h, w, 1)

    Features                        real, positive
    =============================== ==============

    The Data object is a dictionary-like object, with the following attributes
        data: a numpy array of image data of the shape (samples, channels, height, width)
        images: a tuple of the dataset's images
        labels: a numpy array of data's labels containing values from 0 to 4
            0 - bishop
            1 - knight
            2 - pawn
            3 - queen
            4 - rook
        label_names: the names of the data's labels, which are one of the following
            bishop
            knight
            pawn
            queen
            rook
        shape: tuple containing the shape of a single training sample
        unique_labels: a numpy array containing the unique labels
        unique_label_names: a numpy array containing the unique label names

    Additionally, the Data object has methods to resize and grayscale individual images
        resize(image: Image, size: tuple) -> Image
            Returns the image resized to the given size
        grayscale(image: Image) -> Image
            Returns the grayscaled image

    Parameters:
        mode: str, default = 'original'
            The type of data requested. Supported modes are
                'original'
                'resized'
                'grayscaled'
        return_X_y: bool, keyword only, default = False
            If True, returns the data and labels instead of a Data object
        size: tuple, keyword only, default = None
            The size to resize to, only relevant in mode='resized'
        depth_first: bool, keyword onle, default = True:
            Normally, data is returned with the shape (depth, height, width)
            Set depth_first to False to return data with the shape (height, width, depth)

    '''
    # NumPy and PIL (pillow) are required
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError('Module PIL (pillow) is required to load images.')
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError('Module numpy is required to load images.')

    import os

    base = os.path.join(os.path.dirname(__file__), 'images')

    # Root folders
    roots = {
        'original': os.path.join(base, 'original'),
        'grayscaled': os.path.join(base, 'grayscaled'),
        'resized': os.path.join(base, 'resized'),
    }

    # Check root folders exist
    for root in roots.values():
        if not os.path.isdir(root):
            if mode == 'original':
                raise RuntimeError(f'Fatal: subfolder \'original\' not found. Dataset is likely corrupted')
            if mode in ['resized', 'grayscaled']:
                import warnings
                warnings.warn(f'subfolder \'{mode[:-1]}\' not found. Creating \'{root}\' and populating')
                if mode == 'grayscaled':
                    import grayscale
                    grayscale.grayscale()
                if mode == 'resized':
                    import resize
                    resize.resize(size or (100, 100))

    # We label images by their folder
    folders = (
        'bishop',
        'knight',
        'pawn',
        'queen',
        'rook',
    )

    class Data(dict):
        def __init__(self):
            nonlocal folders, roots, size

            # Get root folders for the specified mode
            try:
                root = roots[f'{mode}']
            except KeyError:
                raise ValueError(
                    f'mode={mode} is not a supported mode. Supported modes '
                    'are one of [\'original\', \'resized\', \'grayscaled\']'
                )

            data = []
            images = []
            labels = []
            label_names = []

            for idx, folder in enumerate(folders):
                # Check if subfolder exists
                folder = os.path.join(os.path.join(root, folder))
                if not os.path.isdir(folder):
                    raise RuntimeError(f'Fatal: subfolder \'{folder}\' not found. Dataset is likely corrupted!')

                for image in Data.get_images(folder):
                    if mode == 'resized' or 'resized' in mode:
                        # Resize if required
                        if size:
                            image = Data.resize(image, size)
                    if mode == 'grayscaled' or 'grayscaled' in mode:
                        image = Data.grayscale(image)

                    # PIL Image to NumPy array, of shape (height, width, depth)
                    image_array = np.array(image)

                    # grayscale images do not have depth, so expand dimensions to be uniform
                    if mode == 'grayscaled':
                        image_array = image_array[:, :, np.newaxis]

                    # Convert (height, width, depth) to (depth, height, width) if required
                    if depth_first:
                        image_array = np.transpose(image_array, axes=(-1, 0, 1))  # could also use axes=(2, 0, 1)

                    data.append(image_array)
                    labels.append(idx)
                    images.append(image)
                    label_names.append(folder)

            self['data'] = self.data = np.array(data)
            self['labels'] = self.labels = np.array(labels)
            self['images'] = self.images = tuple(images)  # np.array(images) gives a FutureWarning
            self['label_names'] = self.label_names = np.array(label_names)
            self['shape'] = self.shape = self.data[0].shape
            self['unique_labels'] = self.unique_labels = np.arange(5, dtype=int)
            self['unique_label_names'] = self.unique_label_names = np.array(folders)

        @staticmethod
        def get_files(root):
            '''
            Generator to get filenames of jpg and png files in a folder

            Parameters:
                root: The folder to look for files in

            Returns:
                generator: Yields the jpg and png files in the folder
            '''
            # filter jpg and png images, join filename to the root
            yield from map(lambda x: os.path.join(root, x), filter(lambda x: x.endswith(('.jpg', '.png')), next(os.walk(root))[2]))

        @staticmethod
        def get_images(root):
            '''
            Generator to get jpg and png images as PIL images

            Parameters:
                root: The folder to look for files in

            Returns:
                generator: yields a PIL image for every jpg or png file in the folder
            '''
            # open files as PIL images
            yield from map(Image.open, Data.get_files(root))

        @staticmethod
        def resize(image: Image, size: tuple):
            '''
            Resizes the given image to the given size

            Parameters
                image: Image
                    The PIL image to resize
                size: tuple
                    The size to resize to

            Returns:
                Image: The resized image
            '''
            return image.resize(size)

        @staticmethod
        def grayscale(image):
            '''
            Grayscales the given image to the given size

            Parameters
                image: Image
                    The PIL image to grayscale

            Returns:
                Image: The grayscaled image
            '''
            # PIL.Image's function convert('L') converts image to grayscale
            return image.convert('L')

    data = Data()

    if return_X_y:
        return data.data, data.labels

    return data
