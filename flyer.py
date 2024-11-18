import random
import math
import numpy as np
# import os
def calculate_distance(a=tuple, b=tuple):
    if len(a) != len(b):
        raise ValueError("The length of A and B must be the same")
    return np.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

class Fly:
    def __init__(self, BS_location, flyer_location, distance_set, terminal_threshold=500):
        # self.calculate_distance = A2A_location.calculate_distance
        # self.calculate_loss = A2A_location.calculate_loss
        self.BS_location = BS_location  # dict{BS: location}
        self.flyer_location = flyer_location  # dict{flyer: location}
        self.guess_position = self.state_reset()  # dict{flyer: location}
        self.distance_set = distance_set  # dict{(BS,flyer)/(flyer1,flyer2): distance}
        self.terminal_threshold = terminal_threshold

    def move(self, action, flyer, action_step=50):
        if action == 0:
            self.guess_position[f'flyer_{flyer}'][0] += action_step
        elif action == 1:
            self.guess_position[f'flyer_{flyer}'][0] -= action_step
        elif action == 2:
            self.guess_position[f'flyer_{flyer}'][1] += action_step
        elif action == 3:
            self.guess_position[f'flyer_{flyer}'][1] -= action_step
        elif action == 4:
            self.guess_position[f'flyer_{flyer}'][2] += action_step
        elif action == 5:
            self.guess_position[f'flyer_{flyer}'][2] -= action_step

    def step(self, action):
        flyer = action // 6
        a = action % 6
        terminal = False
        self.move(a, flyer)
        
        # os.system('cls' if os.name == 'nt' else 'clear')
        # for flyer_key in self.flyer_location.keys():
        #     print(f"{flyer_key} - Real Position: {self.flyer_location[flyer_key]}, Estimated Position: {self.guess_position[flyer_key]}, error: {calculate_distance(self.flyer_location[flyer_key], self.guess_position[flyer_key]):6.1f}")

        loss = self.calculate_loss()
        print([float(round(i,2)) for i in loss.values()])
        if sum(loss.values()) / len(loss) <= self.terminal_threshold:
            terminal = True
        # reward = self.cal_reward(loss)
        return self.guess_position, self.cal_reward(loss), terminal
        # return self.guess_position, self.cal_reward_(), terminal

    def calculate_loss(self):
        loss = {}
        for key, distance in self.distance_set.items():
            if key[0] in self.flyer_location.keys():
                place_1 = self.guess_position[key[0]]
            else:
                place_1 = self.BS_location[key[0]]
            place_2 = self.guess_position[key[1]]
            loss[key] = abs(math.sqrt(sum([(place_1[i] - place_2[i]) ** 2 for i in range(3)])) - distance)
        return loss

    def cal_reward(self, loss):
        return 1

    def state_reset(self):
        return {
            flyer_key: [
                int(i + 500 * random.random())
                for i in flyer_position
            ]
            for flyer_key, flyer_position in self.flyer_location.items()
        }


if __name__ == "__main__":
    # import time
    # from A2A_location import A2A_location
    # from A2A_location import calculate_distance
    print('This file do nothing...')