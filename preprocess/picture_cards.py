import os
import argparse
import glob
import random
import string
import numpy as np
import cv2


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default="./data/")
    parser.add_argument('-n', '--region_num', type=int, default=6)

    return parser.parse_args()


def main():
    args = option_parser()

    os.makedirs(args.data_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        src = frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_lower = np.array([0, 0, 190])
        hsv_upper = np.array([255, 255, 255])

        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
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

        k = cv2.waitKey(1)
        if k == ord("q"):
            break

        if k == ord("s"):
            for i in top_idx:
                x0 = stats[i, 0]
                y0 = stats[i, 1]
                x1 = x0 + stats[i, 2]
                y1 = y0 + stats[i, 3]

                rand_name = ''.join([
                    random.choice(string.ascii_letters + string.digits)
                    for _ in range(10)
                ])
                cv2.imwrite(
                    f"{args.data_dir}{rand_name}.jpg", frame[y0:y1, x0:x1]
                )

        cv2.imshow("camera", src)
        cv2.imshow("mask", mask_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
