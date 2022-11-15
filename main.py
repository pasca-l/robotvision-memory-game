import argparse
import cv2

from utils.camera_handle import Camera
from utils.game_info import Game


def option_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_dir', type=str, default="./preprocess/data/"
    )
    parser.add_argument('-n', '--region_num', type=int, default=6)

    return parser.parse_args()


def main():
    args = option_parser()

    camera = Camera(args)
    game = Game(args)

    while True:
        camera.read_frame()
        camera.show_score(game.com_score, game.player_score)
        camera.find_cards()
        print(camera.hold_mark)

        k = cv2.waitKey(1)
        if k == ord("q"):
            break

        if game.end_flag == True:
            camera.show_text("End of game")
            if game.com_score > game.player_score:
                camera.show_result("Computer WIN!")
            elif game.com_score < game.player_score:
                camera.show_result("Player WIN!")
            else:
                camera.show_result("TIE!")
            break

        if k == ord("n"):
            game.mode += 1

            if game.mode == 1 and game.com_turn == True:
                game.choose()
                print("choosed!")
                camera.hold_mark_flag = not camera.hold_mark_flag
                camera.update_hold_mark(game.selection)
                print("marked!")

            if game.mode == 2:
                if game.com_turn == True:
                    camera.hold_mark.clear()
                    camera.hold_mark_flag = not camera.hold_mark_flag
                    print("unmarked!")
                game.remember(camera.region_info, camera.card_num)
                print("memorized!")

                game.judge(camera.region_info, camera.card_num)
                if game.correct_flag == True:
                    camera.card_num -= 2
                    if camera.card_num == 0:
                        game.end_flag = True

            if game.mode == 3:
                if game.correct_flag == True:
                    game.mode = 1
                game.correct_flag = None

            if game.mode == 4:
                game.com_turn = not game.com_turn
                game.mode = 0

        camera.show_text(game.text(game.com_turn, game.mode))
        camera.show_frame()

    camera.cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
