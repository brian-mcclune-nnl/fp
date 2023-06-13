# Flower Photos

This repository explores the
[Load and preprocess
images](https://www.tensorflow.org/tutorials/load_data/images)
demo, incorporating
[DVC](https://dvc.org/).

## Demo steps

1. Run `python get_data.py` to fetch flower photos.
2. Run `for fl in $(ls data/**/*.jpg | awk 'NR%2==0'); do rm $fl; done`
   to delete half the data; this will be to simulate getting more data
   later on.
3. Run `dvc add data/flower_photos`.
4. Run `python get_model.py` to fetch efficientnetb3 without a top.
5. Run `python get_model.py -t` to fetch efficientnetb3 with a top.
6. Run `dvc add models`.
7. Run `dvc run exp`.
8. Demo sections of the DVC extension.
