import sys

if sys.version_info < (3, 6, 0):
    print("neural_network requires python3 version >= 3.6.0", file=sys.stderr)
    sys.exit(1)
