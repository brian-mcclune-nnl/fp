# Flower Photos

This repository explores the
[Load and preprocess
images](https://www.tensorflow.org/tutorials/load_data/images)
demo, incorporating
[DVC](https://dvc.org/).

## Demo steps

1. Run `python scripts/get_data.py` to fetch flower photos.
2. Run `for fl in $(ls data/**/*.jpg | awk 'NR%2==0'); do rm $fl; done`
   to delete half the data; this will be to simulate getting more data
   later on.
3. Run `dvc add data/flower_photos`.
4. Run `python scripts/get_model.py` to fetch efficientnetb0 without a top.
5. Run `dvc add models`.
6. Run `dvc run exp`.
7. Demo sections of the DVC extension.
