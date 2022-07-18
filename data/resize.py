import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

width, height = 108, 135


def process(args):
    root, name = args
    img = Image.open(os.path.join(root, name)).copy()
    img = img.resize((width, height))
    img.save(os.path.join(root, name))


def all_(args):
    files, _root = args
    with ProcessPoolExecutor() as executor:
        results = executor.map(process, ((_root, i) for i in files))
        for result in results:
            pass


if __name__ == '__main__':
    b_root = 'images/resized/bishop/'
    k_root = 'images/resized/knight/'
    p_root = 'images/resized/pawn/'
    q_root = 'images/resized/queen/'
    r_root = 'images/resized/rook/'

    b_files = next(os.walk(b_root))[2]
    k_files = next(os.walk(k_root))[2]
    p_files = next(os.walk(p_root))[2]
    q_files = next(os.walk(q_root))[2]
    r_files = next(os.walk(r_root))[2]
    with ProcessPoolExecutor() as executor:
        results = executor.map(
            all_, (
                (b_files, b_root,),
                (k_files, k_root,),
                (p_files, p_root,),
                (q_files, q_root,),
                (r_files, r_root,),
            )
        )
        for result in results:
            pass
