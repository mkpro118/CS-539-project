def load_data(*, shape=None, grayscale=False):
    from PIL import Image
    import numpy as np
    import os

    root = 'images/original'

    folders = (
        'bishop',
        'knight',
        'pawn',
        'queen',
        'rook',
    )

    class Data:

        def __init__(self):
            nonlocal folders, root, shape, grayscale

            data = []
            images = []
            labels = []
            label_names = []
            for idx, folder in enumerate(folders):
                for image in Data.get_images(os.path.join(root, folder)):
                    if shape:
                        image = Data._resize(image, shape)
                    data.append(np.array(image))
                    images.append(image)
                    labels.append(idx)
                    label_names.append(folder)

            self.data = np.array(data)
            self.images = images
            self.labels = np.array(labels)
            self.label_names = np.array(label_names, dtype=object)

            del data
            del images
            del labels
            del label_names

        @staticmethod
        def get_files(root):
            yield from map(lambda x: os.path.join(root, x), next(os.walk(root))[2])

        @staticmethod
        def get_images(root):
            yield from map(Image.open, Data.get_files(root))

        @staticmethod
        def resize(image, shape):
            return image.resize(shape)

        @staticmethod
        def grayscale(image):
            return image.convert('L')
    return Data()
