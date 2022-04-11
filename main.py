import cv2
import time
import sys

from readonly import Read
from rungame import Game


def main():
    cap = cv2.VideoCapture(0)
    time.sleep(1)

    card_num = int(sys.argv[1])
    camera = Read(cap)
    game = Game(card_num)

    while True:
        camera.show_game(card_num, game.selection)
        camera.show_score(game.com_score, game.player_score)
        data_label = camera.make_label()
        data_label = camera.reshape(data_label)
        camera.inner_conversion()

        if game.end_flag == True:
            camera.show_text("End of game")
            game.mode = 10
            if game.com_score > game.player_score:
                camera.show_result("Computer WIN!")
            elif game.com_score < game.player_score:
                camera.show_result("Player WIN!")
            else:
                camera.show_result("TIE!")

        k = cv2.waitKey(1)
        if k == ord("q"):
            break

        if k == ord("n"):
            game.mode += 1

            if game.mode == 1 and game.com_turn == True:
                game.choose()
                print("choosed!")
                camera.mark_flag = not camera.mark_flag
                camera.hold_mark_flag = not camera.hold_mark_flag
                print("marked!")

            if game.mode == 2:
                if game.com_turn == True:
                    camera.hold_mark.clear()
                    camera.hold_mark_flag = not camera.hold_mark_flag
                    camera.mark_flag = not camera.mark_flag
                    print("unmarked!")
                game.remember(data_label, card_num)
                print("memorized!")
                if game.judge(data_label, card_num):
                    game.correct_flag = True
                    card_num -= 2
                    if card_num == 0:
                        game.end_flag = True
                else:
                    game.correct_flag = False

            if game.mode == 3:
                if game.correct_flag == True:
                    game.forget(game.pick[0])
                    game.correct_flag = None

            if game.mode == 4:
                if game.correct_flag == False:
                    game.com_turn = not game.com_turn
                game.mode = 0

        camera.show_text(game.text(game.com_turn, game.mode))

        cv2.imshow("frame", camera.mark_frame)
        camera.show_mask(data_label)

    camera.cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
