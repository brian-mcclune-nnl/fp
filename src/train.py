"""Train and fine tune a model."""

import csv
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
class AugmentParams:
    """Parameters to configure data augmentation."""
    rotation_factor: float
    translation_factor: float
    contrast_factor: float
    top_dropout_rate: float


@dataclass
class TrainParams:
    """Parameters to configure training."""
    validation_split: float
    batch_size: int
    epochs: int


def train(
    models_params: ModelsParams,
    preprocess_params: PreprocessParams,
    augment_params: AugmentParams,
    train_params: TrainParams,
):
    """Trains a model on the flower photos dataset."""
    # Select model base
    efficientnet_base = models_params.base
    efficientnet_wgts = f'efficientnetb{efficientnet_base}_notop.h5'
    new_model = f'myefficientnetb{efficientnet_base}.h5'

    # Create training and validation dataset splits
    data_path = ROOT_DIR / 'results' / 'data'
    with open(data_path / 'train.txt') as trainf:
        train_photos = trainf.read().splitlines()

    classes = np.unique(
        [pathlib.Path(item).parts[-2] for item in train_photos])
    photos = tf.data.Dataset.from_tensor_slices(train_photos)
    photos = photos.shuffle(len(train_photos), reshuffle_each_iteration=False)

    val_size = int(len(train_photos) * train_params.validation_split)
    train_ds = photos.skip(val_size)
    val_ds = photos.take(val_size)

    def get_label(file_path, classes=classes):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == classes
        return tf.argmax(one_hot)

    def decode_img(img, preprocess_params=preprocess_params):
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(
            img,
            [preprocess_params.img_height, preprocess_params.img_width])

    def process_path(file_path):
        label = get_label(file_path)
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    def configure_for_performance(ds, batch_size=train_params.batch_size):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    # Load the base model
    EfficientNetModel = getattr(
        tf.keras.applications.efficientnet,
        f'EfficientNetB{efficientnet_base}')
    shape = (preprocess_params.img_height, preprocess_params.img_width, 3)
    base_model = EfficientNetModel(
        include_top=False,
        weights=ROOT_DIR / 'models' / efficientnet_wgts,
        input_shape=shape)
    base_model.trainable = False

    # Define augmentation layers
    img_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomRotation(factor=augment_params.rotation_factor),
        tf.keras.layers.RandomTranslation(
            height_factor=augment_params.translation_factor,
            width_factor=augment_params.translation_factor),
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomContrast(factor=augment_params.contrast_factor),
    ])

    # Create trainable model
    inputs = tf.keras.Input(shape=shape)
    x = img_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(
        train_params.top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    # Train and save
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=train_params.epochs)
    models_path = ROOT_DIR / 'results' / 'models'
    models_path.mkdir(parents=True, exist_ok=True)
    model.save(models_path / new_model)

    # Record metrics
    metrics = {
        'loss': history.history['loss'][-1],
        'accuracy': history.history['accuracy'][-1],
    }
    metrics_path = ROOT_DIR / 'results' / 'metrics'
    metrics_path.mkdir(parents=True, exist_ok=True)
    with open(metrics_path / 'train.json', 'w') as metricsf:
        json.dump(metrics, metricsf)

    # Record history
    history_path = ROOT_DIR / 'results' / 'history'
    history_path.mkdir(parents=True, exist_ok=True)
    epochs = range(train_params.epochs)
    columns = ['epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy']
    column_data = zip(epochs, *[history.history[col] for col in columns[1:]])
    with open(history_path / 'train.csv', 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(columns)
        writer.writerows(column_data)


def main():
    """Entry point to train a model on the flower photos dataset."""

    # Read parameters
    all_params = YAML(typ='safe').load(ROOT_DIR / 'params.yaml')
    models_params = ModelsParams(**all_params['models'])
    preprocess_params = TrainParams(**all_params['preprocess'])
    augment_params = TrainParams(**all_params['augment'])
    train_params = TrainParams(**all_params['train'])

    # Train the model
    train(
        models_params,
        preprocess_params,
        augment_params,
        train_params)


if __name__ == '__main__':
    main()
