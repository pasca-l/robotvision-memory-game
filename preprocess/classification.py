import argparse
import json
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.neighbors import NearestNeighbors


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default="./data/")
    parser.add_argument('-n', '--region_num', type=int, default=6)

    return parser.parse_args()


def main():
    args = option_parser()

    with open(f"{args.data_dir}cls2label.json", 'r') as f:
        cls2label = json.load(f)
    label2cls = {v: k for k, v in cls2label.items()}

    cap = cv2.VideoCapture(0)

    features = np.load(f"{args.data_dir}features.npy")
    labels = np.load(f"{args.data_dir}labels.npy")
    model = NearestNeighbors(n_neighbors=1).fit(features)

    while True:
        ret, frame = cap.read()
        cv2.imshow("camera", frame)
        src = frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsvLower = np.array([0, 0, 190])
        hsvUpper = np.array([255, 255, 255])

        mask = cv2.inRange(hsv, hsvLower, hsvUpper)
        mask = cv2.medianBlur(mask, ksize=3)
        mask_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        nlabels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

        if nlabels >= 2:
            top_idx = stats[:, 4].argsort()[-1 - args.region_num:-1]

            for i in top_idx:
                x0 = stats[i, 0]
                y0 = stats[i, 1]
                x1 = x0 + stats[i, 2]
                y1 = y0 + stats[i, 3]

                cv2.rectangle(src, (x0, y0), (x1, y1), (0, 0, 255), 5)

                cv2.putText(
                    src,
                    "Center X: " + str(int(centroids[i, 0])),
                    (x1 - 30, y1 + 15),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    src,
                    "Center Y: " + str(int(centroids[i, 1])),
                    (x1 - 30, y1 + 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    src,
                    "Size: " + str(int(stats[i, 4])),
                    (x1 - 30, y1 + 45),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 255),
                    2,
                )

                gray = cv2.cvtColor(
                    frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY
                )
                gray = cv2.resize(gray, (56, 56))
                feat = hog(gray)
                feat = feat.reshape(1, -1)

                distances, indices = model.kneighbors(feat)

                label = labels[indices[0][0]]
                class_name = label2cls[label]

                cv2.putText(
                    src,
                    "Class: " + str(class_name),
                    (x1 - 30, y1 + 60),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("camera", src)
        cv2.imshow("mask", mask_frame)

        k = cv2.waitKey(1)
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
