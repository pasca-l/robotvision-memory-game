import random

class Game:
    def __init__(self, card_num):
        self.mode = 0
        self.memory = ['back' for i in range(card_num)]
        self.selection = [0, 1]
        self.score = [0, 0]
        self.com_turn = random.choice([True, False])
        self.com_score = 0
        self.player_score = 0
        self.pick = []
        self.correct_flag = None
        self.end_flag = False


    def remember(self, data_label, card_num):
        for i in range(card_num):
            if data_label[2][i] != 'back':
                self.memory[i] = data_label[2][i]


    def forget(self, card):
        self.memory = [i for i in self.memory if i != card]


    def choose(self):
        duplication = [card for card in set(self.memory)
                       if self.memory.count(card) > 1 and card != 'back']
        if duplication:
            for i in duplication:
                self.selection = [j for j, card in enumerate(self.memory)
                                  if card == i]
        else:
            unknown = [i for i, card in enumerate(self.memory) if card == 'back']
            self.selection = random.sample(unknown, 2)

        return self.selection


    def judge(self, data_label, card_num):
        self.pick = []
        for i in range(card_num):
            if data_label[2][i] != "back":
                self.pick.append(data_label[2][i])
        if self.pick[0] == self.pick[1]:
            if self.com_turn == True:
                self.com_score += 2
            else:
                self.player_score += 2
            return True
        return False


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
