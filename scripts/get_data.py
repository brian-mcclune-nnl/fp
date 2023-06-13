"""Fetches flower photos data."""

import argparse
import pathlib
import sys
from typing import List

import tensorflow as tf


def main(argv: List[str] = sys.argv[1:]):
    """Downloads and extracts the flower photos dataset.

    Args:
        argv: Command line arguments.
    """

    parser = argparse.ArgumentParser(description='download flower photos')
    parser.add_argument(
        '-u', '--url',
        default='https://storage.googleapis.com/download.tensorflow.org/'
        'example_images/flower_photos.tgz',
        help='url to download flower photos tarball from')
    parser.add_argument(
        '-n', '--name', default='flower_photos', help='dataset name')
    parser.add_argument(
        '-c', '--cache-subdir', default='data',
        help='location to cache downloaded files')

    args = parser.parse_args(argv)
    data_dir = tf.keras.utils.get_file(
        args.name,
        origin=args.url,
        cache_subdir=args.cache_subdir,
        cache_dir=str(pathlib.Path(__file__).parent.parent),
        untar=True)
    print(f'Flower photos downloaded to: {data_dir}')


if __name__ == '__main__':
    main()
