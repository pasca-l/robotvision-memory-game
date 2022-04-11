import cv2
import numpy as np
import glob
from skimage.feature import hog
from sklearn.neighbors import NearestNeighbors

cap = cv2.VideoCapture(0)

data = glob.glob("./data/*.jpg")
n_data = len(data)

kernel = np.ones((5, 5), np.uint8)

features = np.load("./data/features.npy")
labels = np.load("./data/labels.npy")

model = NearestNeighbors(n_neighbors=1).fit(features)

class_name = None
LABEL2CLS = {0: "background", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "back"}

while True:
    ret, frame = cap.read()
    cv2.imshow("camera", frame)
    src2 = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsvLower = np.array([0, 0, 190])  # 下限
    hsvUpper = np.array([255, 255, 255])  # 上限

    mask = cv2.inRange(hsv, hsvLower, hsvUpper)
    mask = cv2.medianBlur(mask, ksize=3)
    src = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    nlabels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if nlabels >= 2:
        top_idx = stats[:, 4].argsort()[-7:-1]

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

        if class_name is not None:
            display_str = f"class: {class_name}"

        for i in top_idx:
            x0 = stats[i, 0]
            y0 = stats[i, 1]
            x1 = x0 + stats[i, 2]
            y1 = y0 + stats[i, 3]

            gray = cv2.cvtColor(
                frame[y0:y1,x0:x1], cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(gray, (56, 56))
            feat = hog(gray)
            feat = feat.reshape(1, -1)

            distances, indices = model.kneighbors(feat)

            label = labels[indices[0][0]]
            class_name = LABEL2CLS[label]

            cv2.putText(
                src2,
                "Class: " + str(class_name),
                (x1 - 30, y1 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                2,
            )

    cv2.imshow("mask", src)
    cv2.imshow("class", src2)

    k = cv2.waitKey(1)
    if k == ord("q"):
        break

    if k == ord("c"):
        for i in top_idx:
            x0 = stats[i, 0]
            y0 = stats[i, 1]
            x1 = x0 + stats[i, 2]
            y1 = y0 + stats[i, 3]
            cv2.imwrite(
                f"./data/{n_data}.jpg", frame[y0:y1,x0:x1]
            )
            n_data += 1

cap.release()
cv2.destroyAllWindows()
