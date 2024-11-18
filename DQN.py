import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from torch.utils.tensorboard.writer import SummaryWriter
from Maze import Maze

class Net(nn.Module):
    def __init__(self, n_states, n_actions, hide_layers=[15,8], id=None, location=None):
        super(Net, self).__init__()
        if location == None:
            self.location = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        else:
            self.location = location
        self.layers = nn.ModuleList()
        if hide_layers == []:
            self.layers.append(nn.Linear(n_states, n_actions))
        else:
            self.layers.append(nn.Linear(n_states, hide_layers[0]))
            for i in range(len(hide_layers) - 1):
                self.layers.append(nn.Linear(hide_layers[i], hide_layers[i + 1]))
            self.layers.append(nn.Linear(hide_layers[-1], n_actions))
        for i in range(len(self.layers)):
            self.layers[i].weight.data.normal_(0, 0.1)
        # self.fc1 = nn.Linear(n_states, 15)
        # self.fc3 = nn.Linear(15, 8)
        # self.fc2 = nn.Linear(8, n_actions)
        # self.fc1.weight.data.normal_(0, 0.1)
        # self.fc3.weight.data.normal_(0, 0.1)
        # self.fc2.weight.data.normal_(0, 0.1)
        if id == None or self.location == 114514:
            pass
        else:
            with SummaryWriter(log_dir='logs/{}/{}'.format(self.location, id)) as w:
                w.add_graph(self, torch.zeros(1, n_states))
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        out = self.layers[-1](x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # out = self.fc2(x)
        return out


class DQN:
    def __init__(self, n_states, n_actions, hide_layers=[15,8], capacity=800, batch_size=128, location=None):
        print("<DQN init>")
        # DQN有两个net:target net和eval net,具有选动作，存经历，学习三个基本功能
        if location == None:
            self.location = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        else:
            self.location = location
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        if os.path.exists('./logs/{}'.format(self.location)):
            shutil.rmtree('./logs/{}'.format(self.location))
        self.eval_net, self.target_net = Net(n_states, n_actions, hide_layers=hide_layers, location=self.location, id='eval').to(self.device), Net(n_states, n_actions, hide_layers=hide_layers, location=self.location, id='target').to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.n_actions = n_actions
        self.n_states = n_states
        # 使用变量
        self.learn_step_counter = 0  # target网络学习计数
        self.batch_size = batch_size  # batch大小
        self.memory_counter = 0  # 记忆计数
        self.capacity = capacity
        self.memory = np.zeros((self.capacity, self.n_states * 2 + 1 + 1))  # 2*2(state和next_state,每个x,y坐标确定)+2(action和reward),存储2000个记忆体
        self.cost = []  # 记录损失值
        if self.location != 114514:
            self.writer = SummaryWriter(log_dir='logs/{}'.format(self.location))
            self.global_step = 0

    def choose_action(self, x, epsilon):
        # print("<choose_action>")
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)  # (1,2)
        if np.random.uniform() > epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()[0]
        else:
            action = np.random.randint(0, self.n_actions)
        # print("action=", action)
        return action

    def store_transition(self, state, action, reward, next_state):
        # print("<store_transition>")
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.capacity  # 满了就覆盖旧的
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print("<learn>")
        # target net 更新频率,用于预测，不会及时更新参数
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict((self.eval_net.state_dict()))
        self.learn_step_counter += 1

        # 使用记忆库中批量数据
        # sample_index = np.random.choice(min(200, self.memory_counter), 64)  # 2000个中随机抽取32个作为batch_size
        sample_index = np.random.choice(min(self.capacity, self.memory_counter), self.batch_size)  # 2000个中随机抽取32个作为batch_size

        memory = self.memory[sample_index, :]  # 抽取的记忆单元，并逐个提取
        state = torch.FloatTensor(memory[:, :self.n_states]).to(self.device)
        action = torch.LongTensor(memory[:, self.n_states: (self.n_states + 1)]).to(self.device)
        reward = torch.FloatTensor(memory[:, (self.n_states + 1): (self.n_states + 1 + 1)]).to(self.device)
        next_state = torch.FloatTensor(memory[:, (self.n_states + 1 + 1): (self.n_states + 1 + 1 + self.n_states)]).to(self.device)

        # 计算loss,q_eval:所采取动作的预测value,q_target:所采取动作的实际value
        q_eval = self.eval_net(state).gather(1, action) # eval_net->(64,4)->按照action索引提取出q_value
        q_next = self.target_net(next_state).detach()
        # torch.max->[values=[],indices=[]] max(1)[0]->values=[]
        q_target = reward + 0.9 * q_next.max(1)[0].unsqueeze(1) # label
        loss = self.loss(q_eval, q_target)
        self.cost.append(loss)
        # 反向传播更新
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数

        if self.location != 114514:
            self.writer.add_scalar('loss', loss, global_step=self.global_step)
            self.writer.add_scalar('reward', reward.mean(), global_step=self.global_step)
            self.writer.add_histogram('eval_net', self.eval_net(state), global_step=self.global_step)
            for i in range(len(self.eval_net.layers)):
                self.writer.add_histogram(f'fc{i:d}/weights', self.eval_net.layers[i].weight,
                                    global_step=self.global_step)
                self.writer.add_histogram(f'fc{i:d}/biases', self.eval_net.layers[i].bias,
                                    global_step=self.global_step)
            # self.writer.add_histogram(f'fc{i:d}/weights', self.eval_net.fc1.weight,
            #                         global_step=self.global_step)
            # self.writer.add_histogram('fc1/biases', self.eval_net.fc1.bias,
            #                         global_step=self.global_step)
            # self.writer.add_histogram('fc3/weights', self.eval_net.fc3.weight,
            #                         global_step=self.global_step)
            # self.writer.add_histogram('fc3/biases', self.eval_net.fc3.bias,
            #                         global_step=self.global_step)
            # self.writer.add_histogram('fc2/weights', self.eval_net.fc2.weight,
            #                         global_step=self.global_step)
            # self.writer.add_histogram('fc2/biases', self.eval_net.fc2.bias,
            #                         global_step=self.global_step)
            self.global_step += 1

    def plot_cost(self):
        # 检查 self.cost 是否包含多个 Tensor，并将所有 Tensor 移动到 CPU，移除梯度信息
        if isinstance(self.cost, list):
            # 将列表中的每个 Tensor 移动到 CPU，使用 detach() 移除梯度信息，并转换为 NumPy 数组
            cost_cpu = [
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) and x.device.type in ('cuda', 'mps') else x for
                x in self.cost]
        elif isinstance(self.cost, torch.Tensor):
            # 如果 self.cost 是单个 Tensor，则直接处理并移除梯度信息
            cost_cpu = self.cost.detach().cpu().numpy() if self.cost.device.type in (
            'cuda', 'mps') else self.cost.detach().numpy()
        else:
            # 如果 self.cost 不是 Tensor，则直接转换
            cost_cpu = np.array(self.cost)

        # 转换为平坦的 NumPy 数组，以便 matplotlib 正常处理
        cost_cpu = np.array(cost_cpu).flatten()

        plt.plot(np.arange(len(cost_cpu)), cost_cpu)
        plt.xlabel("step")
        plt.ylabel("cost")
        plt.show()


if __name__ == "__main__":
    import time


    def run_maze():
        print("====Game Start====")
        step = 0
        max_episode = 500
        for episode in range(max_episode):
            state = env.reset()  # 重置智能体位置
            step_every_episode = 0
            epsilon = episode / max_episode  # 动态变化随机值
            while True:
                if episode < 20:
                    time.sleep(0.1)
                if episode > max_episode-10:
                    time.sleep(0.5)
                env.render()  # 显示新位置
                action = model.choose_action(state, 1-epsilon)  # 根据状态选择行为
                # 环境根据行为给出下一个状态，奖励，是否结束。
                next_state, reward, terminal = env.step(action)
                # reward -= 1/400*step_every_episode**2
                model.store_transition(state, action, reward, next_state)  # 模型存储经历
                # 控制学习起始时间(先积累记忆再学习)和控制学习的频率(积累多少步经验学习一次)
                if step > 200 and step % 5 == 0:
                    model.learn()
                # 进入下一步
                state = next_state
                if terminal or step_every_episode > 99:
                    print("episode=", episode, end=",")
                    print("step=", step_every_episode)
                    break
                step += 1
                step_every_episode += 1
        # 游戏环境结束
        print("====Game Over====")
        env.destroy()

    env = Maze()  # 环境
    model = DQN(
        n_states=env.n_states,
        n_actions=env.n_actions,
        capacity=200,
        batch_size=64,
        location=114514
    )  # 算法模型
    run_maze()
    env.mainloop()
    model.plot_cost()  # 误差曲线