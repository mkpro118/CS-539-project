'''
Utility functions to grayscale all images in the 'images/original/' folder
'''

import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor


def process(args):
    '''
    Grayscales an image, and saves them to the corresponding folder

    Parameters:
        args: tuple
            The tuple of the original folder, new folder, filename

    Note: Takes in a single argument instead of 3
          to allow the usage of ProcessPoolExecutor
    '''
    # unpack args
    original, grayscaled, name = args

    # grayscale
    img = Image.open(os.path.join(original, name)).convert('L')

    # save
    img.save(os.path.join(grayscaled, name))


def all_(args):
    '''
    Grayscales all images in the given subfolder

    Parameters:
        args: tuple
            The tuple of the original filename, original folder, new folder

    Note: Takes in a single argument instead of 3
          to allow the usage of ProcessPoolExecutor
    '''

    files, _root, _new = args
    with ProcessPoolExecutor() as executor:
        results = executor.map(process, ((_root, _new, i) for i in files))
        for result in results:
            pass


def grayscale():
    '''
    Primary function in this module, grayscales all images found in 'images/original/'
    '''
    # tuple of original image folders
    original_folders = (
        'images/original/bishop/',
        'images/original/knight/',
        'images/original/pawn/',
        'images/original/queen/',
        'images/original/rook/',
    )

    # tuple of grayscaled image folders
    new_folders = (
        'images/grayscaled/bishop/',
        'images/grayscaled/knight/',
        'images/grayscaled/pawn/',
        'images/grayscaled/queen/',
        'images/grayscaled/rook/',
    )

    # Ensure original folders exist
    for folder in original_folders:
        assert os.path.isdir(folder)

    # Create new folders if needed
    for new_folder in new_folders:
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)

    # get tuple of all files
    files = tuple((next(os.walk(folder))[2] for folder in original_folders))

    # grayscale with multiprocessing to speed things up
    with ProcessPoolExecutor() as executor:
        results = executor.map(all_, (zip(files, original_folders, new_folders)))
        for result in results:
            pass


if __name__ == '__main__':
    grayscale()
