"""Splits flowers dataset into train and test."""

import pathlib
from dataclasses import dataclass

import tensorflow as tf
from ruamel.yaml import YAML

ROOT_DIR = pathlib.Path(__file__).parent.parent


@dataclass
class SplitParams:
    """Parameters to configure splitting."""
    test_split: float


def split(params: SplitParams):
    """Splits flower photos dataset into train and test.

    """
    # Create training and test dataset splits
    data_dir = ROOT_DIR / 'data' / 'flower_photos'
    image_count = len(list(data_dir.glob('*/*.jpg')))
    photos = tf.data.Dataset.list_files(
        str(data_dir / '*/*.jpg'), shuffle=False)
    photos = photos.shuffle(image_count, reshuffle_each_iteration=False)

    test_size = int(image_count * params.test_split)
    train_photos = photos.skip(test_size)
    test_photos = photos.take(test_size)

    # Record splits to .txt data files
    data_path = ROOT_DIR / 'results' / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
    with open(data_path / 'train.txt', 'w') as trainf:
        for photo in train_photos:
            trainf.write(f'{photo.numpy().decode()}\n')
    with open(data_path / 'test.txt', 'w') as testf:
        for photo in test_photos:
            testf.write(f'{photo.numpy().decode()}\n')


def main():
    """Entry point to split flower photos dataset into train and test.

    """
    # Read training parameters
    all_params = YAML(typ='safe').load(ROOT_DIR / 'params.yaml')
    params = SplitParams(**all_params['split'])

    # Split the dataset
    split(params)


if __name__ == '__main__':
    main()
