import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor


def process(args):
    original, resized, name = args
    img = Image.open(os.path.join(original, name)).copy()
    img = img.convert('L')
    img.save(os.path.join(resized, name))


def all_(args):
    files, _root, _new = args
    with ProcessPoolExecutor() as executor:
        results = executor.map(process, ((_root, _new, i) for i in files))
        for result in results:
            pass


if __name__ == '__main__':
    b_root = 'images/original/bishop/'
    k_root = 'images/original/knight/'
    p_root = 'images/original/pawn/'
    q_root = 'images/original/queen/'
    r_root = 'images/original/rook/'

    b_new = 'images/resized/bishop/'
    k_new = 'images/resized/knight/'
    p_new = 'images/resized/pawn/'
    q_new = 'images/resized/queen/'
    r_new = 'images/resized/rook/'

    b_files = next(os.walk(b_root))[2]
    k_files = next(os.walk(k_root))[2]
    p_files = next(os.walk(p_root))[2]
    q_files = next(os.walk(q_root))[2]
    r_files = next(os.walk(r_root))[2]
    with ProcessPoolExecutor() as executor:
        results = executor.map(
            all_, (
                (b_files, b_root, b_new),
                (k_files, k_root, k_new),
                (p_files, p_root, p_new),
                (q_files, q_root, q_new),
                (r_files, r_root, r_new),
            )
        )
        for result in results:
            pass
