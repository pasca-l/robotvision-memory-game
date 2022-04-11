import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import NearestNeighbors

class Read:
    def __init__(self, cap):
        self.cap = cap
        self.cap_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cap_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame = None
        self.mark_frame = None
        self.kernel = np.ones((5, 5), np.uint8)
        self.features = np.load("./data/features.npy")
        self.labels = np.load("./data/labels.npy")
        self.model = NearestNeighbors(n_neighbors=1).fit(self.features)
        self.class_name = None
        self.LABEL2CLS = {0: "background", 1: "1", 2: "2", 3: "3", 4: "4",
                        5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10",
                        11: "back"}
        self.data_label = []
        self.mark_flag = False
        self.coordinate_flag = False
        self.mask = None
        self.card_num = None
        self.coordinates = []
        self.hold_mark_flag = False
        self.hold_mark = []


    def show_game(self, card_num, selection):
        ret, self.frame = self.cap.read()
        try:
            self.mark_frame = self.frame.copy()
        except:
            pass
        self.card_num = card_num

        if self.mark_flag == True:
            self.mark(self.data_label, selection)


    def make_label(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        hsvLower = np.array([0, 0, 195])
        hsvUpper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, hsvLower, hsvUpper)
        self.mask = cv2.medianBlur(mask, ksize=3)

        nlabels, _, stats, centroids = cv2.connectedComponentsWithStats(self.mask)

        if nlabels >= 2:
            self.data_label = []

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
                self.class_name = self.LABEL2CLS[label]

                self.data_label.append([int(centroids[i, 0]),
                                        int(centroids[i, 1]),
                                        self.class_name,
                                        stats[i, 0], stats[i, 1],
                                        stats[i, 2], stats[i, 3]])

            self.data_label.sort(reverse=True, key=lambda x: x[0] + x[1])

        return self.data_label


    def reshape(self, labels):
        x_center = []
        y_center = []
        class_name = []
        box = []

        for _, label in enumerate(labels):
            x_center.append(label[0])
            y_center.append(label[1])
            class_name.append(label[2])
            box.append([label[3], label[4], label[5], label[6]])

        return [x_center, y_center, class_name, box]


    def inner_conversion(self):
        self.data_label = self.reshape(self.data_label)


    def show_mask(self, data_label):
        src = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        for i, _ in enumerate(data_label[2]):
            x0 = data_label[3][i][0]
            y0 = data_label[3][i][1]
            x1 = x0 + data_label[3][i][2]
            y1 = y0 + data_label[3][i][3]

            cv2.rectangle(src, (x0, y0), (x1, y1), (0, 0, 255), 5)
            cv2.putText(
                src,
                "Class: " + data_label[2][i],
                (x1 - 30, y1 + 15),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 255),
                2,
            )

        cv2.imshow("mask", src)


    def mark(self, data_label, selection):
        if self.hold_mark:
            for i, _ in enumerate(self.hold_mark):
                cv2.rectangle(self.mark_frame,
                              (self.hold_mark[i][0], self.hold_mark[i][1]),
                              (self.hold_mark[i][2], self.hold_mark[i][3]),
                              (0, 0, 255), 5)
            return

        for i in selection:
            x0 = data_label[3][i][0]
            y0 = data_label[3][i][1]
            x1 = x0 + data_label[3][i][2]
            y1 = y0 + data_label[3][i][3]

            if self.hold_mark_flag == True:
                self.hold_mark.append([x0, y0, x1, y1])
            else:
                cv2.rectangle(self.mark_frame, (x0, y0), (x1, y1), (0, 0, 255), 5)


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
