import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor


def process(args):
    original, grayscaled, name = args
    img = Image.open(os.path.join(original, name)).copy()
    img = img.convert('L')
    img.save(os.path.join(grayscaled, name))


def all_(args):
    files, _root, _new = args
    with ProcessPoolExecutor() as executor:
        results = executor.map(process, ((_root, _new, i) for i in files))
        for result in results:
            pass


def grayscale():
    original_folders = (
        'images/original/bishop/',
        'images/original/knight/',
        'images/original/pawn/',
        'images/original/queen/',
        'images/original/rook/',
    )

    new_folders = (
        'images/grayscaled/bishop/',
        'images/grayscaled/knight/',
        'images/grayscaled/pawn/',
        'images/grayscaled/queen/',
        'images/grayscaled/rook/',
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
    grayscale()
