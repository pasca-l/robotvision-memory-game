import json
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import NearestNeighbors


class Camera():
    def __init__(self, args):
        self.card_num = args.region_num
        with open(f"{args.data_dir}cls2label.json", 'r') as f:
            cls2label = json.load(f)
        self.label2cls = {v: k for k, v in cls2label.items()}

        self.features = np.load(f"{args.data_dir}features.npy")
        self.labels = np.load(f"{args.data_dir}labels.npy")
        self.model = NearestNeighbors(n_neighbors=1).fit(self.features)

        self.cap = cv2.VideoCapture(0)
        self.cap_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cap_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.frame = None
        self.mark_frame = None
        self.region_info = []
        self.mask = None
        self.hold_mark_flag = False
        self.hold_mark = []

    def read_frame(self):
        ret, self.frame = self.cap.read()
        self.mark_frame = self.frame.copy()

    def show_frame(self):
        if self.hold_mark_flag:
            for i, _ in enumerate(self.hold_mark):
                cv2.rectangle(self.mark_frame,
                              (self.hold_mark[i][0], self.hold_mark[i][1]),
                              (self.hold_mark[i][2], self.hold_mark[i][3]),
                              (0, 0, 255), 5)

        cv2.imshow("frame", self.mark_frame)

        src = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        for i, _ in enumerate(self.region_info[2]):
            x0 = self.region_info[3][i][0]
            y0 = self.region_info[3][i][1]
            x1 = x0 + self.region_info[3][i][2]
            y1 = y0 + self.region_info[3][i][3]

            cv2.rectangle(src, (x0, y0), (x1, y1), (0, 0, 255), 5)
            cv2.putText(
                src,
                "Class: " + self.region_info[2][i],
                (x1 - 30, y1 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                2,
            )

        cv2.imshow("mask", src)

    def find_cards(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        hsv_lower = np.array([0, 0, 195])
        hsv_upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        self.mask = cv2.medianBlur(mask, ksize=3)

        nlabels, _, stats, centroids = cv2.connectedComponentsWithStats(self.mask)

        box_info = []
        if nlabels >= 2:
            self.region_info = []

            top_idx = stats[:, 4].argsort()[-1*self.card_num -1:-1]
            for i in top_idx:
                x0 = stats[i, 0]
                y0 = stats[i, 1]
                x1 = x0 + stats[i, 2]
                y1 = y0 + stats[i, 3]

                gray = cv2.cvtColor(self.frame[y0:y1,x0:x1], cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (56, 56))

                feat = hog(gray)
                feat = feat.reshape(1, -1)
                distances, indices = self.model.kneighbors(feat)

                label = self.labels[indices[0][0]]
                class_name = self.label2cls[label]

                box_info.append([
                    int(centroids[i, 0]), int(centroids[i, 1]),
                    class_name,
                    stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
                ])

            box_info.sort(reverse=True, key=lambda x: x[0] + x[1])

        x_center = []
        y_center = []
        class_name = []
        box = []

        for _, label in enumerate(box_info):
            x_center.append(label[0])
            y_center.append(label[1])
            class_name.append(label[2])
            box.append([label[3], label[4], label[5], label[6]])

        self.region_info = [x_center, y_center, class_name, box]

        return self.region_info

    def show_text(self, text):
        cv2.putText(
            self.mark_frame,
            text,
            (50, 50),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 0),
            2,
        )

    def show_score(self, com_score, player_score):
        cv2.putText(
            self.mark_frame,
            "Computer : " + str(com_score),
            (int(self.cap_width) - 250, int(self.cap_height) - 50),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            self.mark_frame,
            "Player : " + str(player_score),
            (50, int(self.cap_height) - 50),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 0),
            2,
        )

    def show_result(self, text):
        cv2.putText(
            self.mark_frame,
            text,
            (200, 300),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 255, 0),
            2,
        )

    def update_hold_mark(self, selection):
        for i in selection:
            x0 = self.region_info[3][i][0]
            y0 = self.region_info[3][i][1]
            x1 = x0 + self.region_info[3][i][2]
            y1 = y0 + self.region_info[3][i][3]

            self.hold_mark.append([x0, y0, x1, y1])
