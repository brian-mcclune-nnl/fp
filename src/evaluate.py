"""Evaluate a model."""

import json
import os
import pathlib
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from ruamel.yaml import YAML

ROOT_DIR = pathlib.Path(__file__).parent.parent


@dataclass
class ModelsParams:
    """Parameters to configure model selection."""
    base: int


@dataclass
class PreprocessParams:
    """Parameters to configure preprocessing."""
    img_height: int
    img_width: int


@dataclass
class EvaluateParams:
    """Parameters to configure evaluation."""
    batch_size: float


def evaluate(
    models_params: ModelsParams,
    preprocess_params: PreprocessParams,
    evaluate_params: EvaluateParams,
):
    """Evaluates a model on the flower photos dataset.

    Args:
        models_params: Model-specific config parameters.
        train_params: Training-specific config parameters.
        evaluate_params: Evaluation-specific config parameters.
    """
    efficientnet_base = models_params.base
    new_model = f'myefficientnetb{efficientnet_base}.h5'

    # Create test dataset
    data_path = ROOT_DIR / 'results' / 'data'
    with open(data_path / 'test.txt') as testf:
        test_photos = testf.read().splitlines()

    classes = np.unique(
        [pathlib.Path(item).parts[-2] for item in test_photos])
    photos = tf.data.Dataset.from_tensor_slices(test_photos)
    photos = photos.shuffle(len(test_photos), reshuffle_each_iteration=False)

    def get_label(file_path, classes=classes):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == classes
        return tf.argmax(one_hot)

    def decode_img(img):
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(
            img,
            [preprocess_params.img_height, preprocess_params.img_width])

    def process_path(file_path):
        label = get_label(file_path)
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    test_ds = photos.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    def configure_for_performance(ds, batch_size=evaluate_params.batch_size):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    test_ds = configure_for_performance(test_ds)

    # Load the model
    model = tf.keras.models.load_model(
        ROOT_DIR / 'results' / 'models' / new_model)

    # Evaluate the model and record metrics
    loss, accuracy = model.evaluate(test_ds)
    metrics = {'loss': loss, 'accuracy': accuracy}
    metrics_path = ROOT_DIR / 'results' / 'metrics'
    metrics_path.mkdir(parents=True, exist_ok=True)
    with open(metrics_path / 'evaluate.json', 'w') as metricsf:
        json.dump(metrics, metricsf)


def main():
    """Entry point to evaluate a model on the flower photos dataset."""
    # Read parameters
    all_params = YAML(typ='safe').load(ROOT_DIR / 'params.yaml')
    models_params = ModelsParams(**all_params['models'])
    preprocess_params = ModelsParams(**all_params['preprocess'])
    evaluate_params = EvaluateParams(**all_params['evaluate'])

    # Evaluate the model
    evaluate(
        models_params,
        preprocess_params,
        evaluate_params)


if __name__ == '__main__':
    main()
