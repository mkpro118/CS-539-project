import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

width, height = 224, 244


def process(args):
    original, resized, name = args
    img = Image.open(os.path.join(original, name)).copy()
    img = img.resize((width, height))
    img.save(os.path.join(resized, name))


def all_(args):
    files, _root, _new = args
    with ProcessPoolExecutor() as executor:
        results = executor.map(process, ((_root, _new, i) for i in files))
        for result in results:
            pass


def resize(shape):
    nonlocal width, height

    width, height = shape

    original_folders = (
        'images/original/bishop/',
        'images/original/knight/',
        'images/original/pawn/',
        'images/original/queen/',
        'images/original/rook/',
    )

    new_folders = (
        'images/resized/bishop/',
        'images/resized/knight/',
        'images/resized/pawn/',
        'images/resized/queen/',
        'images/resized/rook/',
    )

    for folder in original_folders:
        assert os.path.isdir(folder)

    for new_folder in new_folders:
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)

    files = tuple((next(os.walk(folder))[2] for folder in original_folders))

    with ProcessPoolExecutor() as executor:
        results = executor.map(all_, (zip(files, original_folders, new_folders)))
        for result in results:
            pass


if __name__ == '__main__':
    resize(width, height)
