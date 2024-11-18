from DQN import DQN
from flyer import calculate_distance
from flyer import Fly


# import math

if __name__ == '__main__':
    BS = {
        'BS_0': (0, 0, 0),
        'BS_1': (100000, 100000, 0),
        'BS_2': (0, 100000, 0),
        'BS_3': (100000, 0, 0),
    }

    flyers = {
        'flyer_0': (10000, 10000, 30000),
        'flyer_1': (70000, 70000, 70000),
        'flyer_2': (10000, 60000, 10000),
        'flyer_3': (10000, 10000, 70000),
        'flyer_4': (50000, 50000, 50000),
    }

    A2A_distance = {
        (flyer_key, flyer_key_): calculate_distance(flyer_position, flyer_position_)
        for flyer_key, flyer_position in flyers.items()
        for flyer_key_, flyer_position_ in flyers.items()
        if flyer_key != flyer_key_ and flyer_key < flyer_key_
    }

    A2G_distance = {
        ('BS_0', 'flyer_0'): calculate_distance(BS['BS_0'], flyers['flyer_0']),
        ('BS_1', 'flyer_1'): calculate_distance(BS['BS_1'], flyers['flyer_1']),
        ('BS_2', 'flyer_2'): calculate_distance(BS['BS_2'], flyers['flyer_2']),
        ('BS_3', 'flyer_3'): calculate_distance(BS['BS_3'], flyers['flyer_3']),
    }

    # All_distance = A2A_distance | A2G_distance

    env = Fly(
        BS_location=BS,
        flyer_location=flyers,
        distance_set=A2A_distance | A2G_distance,
    )  # 环境

    model = DQN(
        n_states=3 * len(flyers),
        n_actions=6 * len(flyers),
        hide_layers=[32, 64, 128, 128, 64, 32],
        location=None,
    )  # 算法模型

    print("====Training Start====")
    step = 0
    max_episode = 500000000
    for episode in range(max_episode):
        epsilon = episode / max_episode
        terminal = False
        episode_reward = 0
        ep_step = 0
        while not terminal:
            state = []
            for i in range(len(flyers)):
                state += env.guess_position[f'flyer_{i}']
            action = model.choose_action(state, 1 - epsilon)
            next_state, reward, terminal = env.step(action)
            next_state = []
            for i in range(len(flyers)):
                next_state += env.guess_position[f'flyer_{i}']
            model.store_transition(state, action, reward, next_state)
            episode_reward += reward
            if step > 200:
                model.learn()
            if ep_step > 15000:
                terminal = True
            ep_step += 1
            step += 1
        print(f"Episode: {episode}, Reward: {episode_reward / ep_step:.2f}")
