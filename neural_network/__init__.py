import sys

if sys.version_info < (3, 9, 0):
    print("neural_network requires python3 version >= 3.9.0", file=sys.stderr)
