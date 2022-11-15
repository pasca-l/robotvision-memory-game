import argparse
import glob
import json
import numpy as np
import cv2
from skimage.feature import hog


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default="./data/")

    return parser.parse_args()


def main():
    args = option_parser()
    classes = [i.split('/')[-2] for i in glob.glob(f"{args.data_dir}*/")]
    cls2label = {i: n for n, i in enumerate(classes)}
    with open(f"{args.data_dir}cls2label.json", 'w') as f:
        json.dump(cls2label, f)

    features, labels = [], []

    for c in classes:
        label = cls2label[c]

        image_paths = glob.glob(f"{args.data_dir}{c}/*.jpg")
        for path in image_paths:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (56, 56))

            feat = hog(gray)
            features.append(feat)
            labels.append(label)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.uint8)

    np.save(f"{args.data_dir}features.npy", features)
    np.save(f"{args.data_dir}labels.npy", labels)


if __name__ == '__main__':
    main()
