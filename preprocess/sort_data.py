import os
import argparse
import glob
import shutil
import cv2


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default="./data/")

    return parser.parse_args()


def main():
    args = option_parser()
    data = glob.glob(f"{args.data_dir}*.jpg")

    key_dict = {
        ord("1"): 1,
        ord("2"): 2,
        ord("3"): 3,
        ord("4"): 4,
        ord("5"): 5,
        ord("6"): 6,
        ord("7"): 7,
        ord("8"): 8,
        ord("9"): 9,
        ord("0"): 10,
        ord("b"): "back"
    }

    for i in data:
        image = cv2.imread(i)
        cv2.imshow(f"{i}", image)

        k = cv2.waitKey(0)
        if k == ord("q"):
            cv2.destroyAllWindows()
            return

        if k in key_dict.keys():
            sort_dir = f"{args.data_dir}{key_dict[k]}"
            os.makedirs(sort_dir, exist_ok=True)
            shutil.move(i, sort_dir)
            print(f"Moved {i} to {sort_dir}")
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
