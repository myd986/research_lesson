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
    def __init__(self, n_states, n_actions, id=None, location=None):
        super(Net, self).__init__()
        if location == None:
            self.location = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        else:
            self.location = location
        self.fc1 = nn.Linear(n_states, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        if id == None or self.location == 114514:
            pass
        else:
            with SummaryWriter(log_dir='logs/{}/{}'.format(self.location, id)) as w:
                w.add_graph(self, torch.zeros(1, n_states))
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


class DQN:
    def __init__(self, n_states, n_actions, capacity=800, batch_size=128, location=None):
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
        self.eval_net, self.target_net = Net(n_states, n_actions, location=self.location, id='eval').to(self.device), Net(n_states, n_actions, location=self.location, id='target').to(self.device)
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
            self.target_net.load_state_dict(self.eval_net.state_dict())
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

            self.writer.add_histogram('fc1/weights', self.eval_net.fc1.weight,
                                    global_step=self.global_step)
            self.writer.add_histogram('fc1/biases', self.eval_net.fc1.bias,
                                    global_step=self.global_step)
            self.writer.add_histogram('fc3/weights', self.eval_net.fc3.weight,
                                    global_step=self.global_step)
            self.writer.add_histogram('fc3/biases', self.eval_net.fc3.bias,
                                    global_step=self.global_step)
            self.writer.add_histogram('fc2/weights', self.eval_net.fc2.weight,
                                    global_step=self.global_step)
            self.writer.add_histogram('fc2/biases', self.eval_net.fc2.bias,
                                    global_step=self.global_step)
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
    pass