import random


class Game():
    def __init__(self, args):
        self.mode = 0
        self.memory = ['back' for i in range(args.region_num)]
        self.selection = [0, 1]
        self.com_turn = True #random.choice([True, False])
        self.com_score = 0
        self.player_score = 0
        self.pick = []
        self.correct_flag = None
        self.end_flag = False

    def remember(self, region_info, card_num):
        for i in range(card_num):
            if region_info[2][i] != 'back':
                self.memory[i] = region_info[2][i]

    def choose(self):
        duplication = [
            card for card in set(self.memory)
            if self.memory.count(card) > 1 and card != 'back'
        ]
        if duplication:
            for i in duplication:
                self.selection = [
                    j for j, card in enumerate(self.memory)if card == i
                ]
        else:
            unknown = [
                i for i, card in enumerate(self.memory) if card == 'back'
            ]
            self.selection = random.sample(unknown, 2)

    def judge(self, region_info, card_num):
        pick = []
        for i in range(card_num):
            if region_info[2][i] != "back":
                pick.append(i)
        if region_info[2][pick[0]] == region_info[2][pick[1]]:
            self._forget(pick)
            if self.com_turn == True:
                self.com_score += 2
            else:
                self.player_score += 2

            self.correct_flag = True
        else:
            self.correct_flag = False

    def _forget(self, picks):
        self.memory.pop(picks[0])
        self.memory.pop(picks[1])

    def text(self, turn, mode):
        if turn == True:
            if mode == 0:
                return "Computer : standby"
            elif mode == 1:
                return "Computer : pick and show"
            elif mode == 2:
                if self.correct_flag == True:
                    return "Computer : correct selection, take away cards"
                elif self.correct_flag == False:
                    return "Computer : wrong selection, return cards"
            elif mode == 3:
                return "Computer : changing player"
        elif turn == False:
            if mode == 0:
                return "Player : standby"
            elif mode == 1:
                return "Player : pick and show"
            elif mode == 2:
                if self.correct_flag == True:
                    return "Player : correct selection, take away cards"
                elif self.correct_flag == False:
                    return "Player : wrong selection, turn cards"
            elif mode == 3:
                return "Player : changing player"
