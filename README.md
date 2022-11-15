# Robot Vision Memory Game

## Screenshots

## Requirements
- opencv-python 4.6.0.66
- scikit-learn 1.1.3
- scikit-image 0.19.3

## Usage
0. Calculate the HOG features using scripts under `preprocess` in the following order.
   1. `picture_cards.py`
   2. `sort_data.py`
   3. `calc_hog.py`
   4. `classification.py` (optional)
1. Run `main.py` to start the game, giving options below:
    - `-d` or `--data_dir` for designating the name of the data folder containing `cls2label.json`, `features.npz`, `labels.npz`, which will be generated from step 0.
    - `-n` or `--region_num` for designating the number of bounding boxes to detect. (Equivalent to the number of cards that will be used).
```shell
$ python main.py [-d DATA_DIRECTORY] [-n REGION_NUM]
```
2. Instructions will be given at the top left on the screen. To move on press `n` key.
    - Pressing the `q` key will exit at any time.

## Note
The HSV value for finding the masks of the cards is set to a constant. Therefore, depending on the environment, such as lighting, this will affect the detection.