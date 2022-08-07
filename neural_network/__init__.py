import sys

if sys.version_info < (3, 6, 0):
    print("neural_network requires python3 version >= 3.6.0", file=sys.stderr)
    sys.exit(1)
elif sys.version_info < (3, 8, 0):
    import warnings
    x, y, z = sys.version_info[:3]
    warnings.warn('typing support is not compatible with python version ' + '.'.join((x, y, z)))
