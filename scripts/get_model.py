"""Gets model weights."""

import argparse
import pathlib
import sys
from typing import List

import tensorflow as tf


def main(argv: List[str] = sys.argv[1:]):
    """Downloads model weights.

    Args:
        argv: Command line arguments.
    """

    parser = argparse.ArgumentParser(description='download efficientnet')
    parser.add_argument(
        '-u', '--url',
        default='https://storage.googleapis.com/keras-applications/',
        help='base url to download models from')
    parser.add_argument(
        '-n', '--name', default='efficientnetb0', help='model name')
    parser.add_argument(
        '-c', '--cache-subdir', default='models',
        help='location to cache downloaded files')
    parser.add_argument(
        '-t', '--include-top', action='store_true',
        help='include fully connected layer at top of network')

    args = parser.parse_args(argv)
    url = args.url + args.name + ('.h5' if args.include_top else '_notop.h5')
    data_dir = tf.keras.utils.get_file(
        args.name + ('.h5' if args.include_top else '_notop.h5'),
        origin=url,
        cache_subdir=args.cache_subdir,
        cache_dir=str(pathlib.Path(__file__).parent.parent))
    print(f'Model downloaded to: {data_dir}')


if __name__ == '__main__':
    main()
