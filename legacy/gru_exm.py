import os

import numpy as np
import cv2

import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
from torch.nn import GRU
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm
import pandas as pd
import scipy.signal
from scipy.interpolate import CubicSpline

from scipy.signal import savgol_filter
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class PulseDataset(Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


def build_dataloader(data, label, batch_size=8):
    dataset = PulseDataset(data, label)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# 构建GRU
class MyGru(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size=256, output_size=1, num_layers=1):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x):
        output, _ = self.gru(x, None)
        output = output[:, -1, :]
        output = self.linear(output)
        return output


class CNN_GRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, output_size=11, num_layers=1):
        super().__init__()
        # CNN 部分：压缩序列长度
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            #nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2),

            #nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=2), # 序列长度减半：1500 → 750
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # GRU 部分：处理压缩后的序列
        self.gru = nn.GRU(
            input_size=64,  # CNN 输出的通道数作为 GRU 的输入维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # 输出层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入 x 形状: [batch_size, seq_len=1500, input_size=3]

        # Step 1: CNN 处理
        x = x.permute(0, 2, 1)  # 转为 [batch, input_size, seq_len] 适应 Conv1d
        x = self.cnn(x)  # 输出 [batch, 64, seq_len//2=750]
        x = x.permute(0, 2, 1)  # 恢复为 [batch, seq_len//2=750, 64]

        # Step 2: GRU 处理
        output, _ = self.gru(x)  # output: [batch, 750, hidden_size]
        output = output[:, -1, :]  # 取最后一个时间步 [batch, hidden_size]

        # Step 3: 输出预测
        output = self.linear(output)
        return output

'''
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        attn_weights = torch.softmax(self.attention(x), dim=1)  # [batch, seq_len, 1]
        return (x * attn_weights).sum(dim=1)  # 加权求和 → [batch, hidden_size]


class CNN_GRU2(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, output_size=11):
        super().__init__()
        self.cnn = nn.Sequential(
            # 第一层：大幅降低序列长度
            nn.Conv1d(input_size, 64, kernel_size=7, stride=2, padding=2),  # 1500 → 500
            nn.ReLU(),
            nn.BatchNorm1d(64),  # 加速收敛

            # 第二层：进一步压缩
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=1),  # 500 → 250
            nn.ReLU(),
            nn.BatchNorm1d(128),

            # 第三层：最终压缩
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # 250 → 125
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        # 双向GRU
        self.gru = nn.GRU(
            input_size=256,  # CNN输出的通道数
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.attention = Attention(hidden_size)  # 双向GRU输出维度翻倍
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)  # [batch, 256, 125]
        x = x.permute(0, 2, 1)  # [batch, 125, 256]

        output, _ = self.gru(x)  # [batch, 125, hidden_size*2]
        output = self.attention(output)  # [batch, hidden_size*2]
        return self.linear(output)
'''
'''
def class_emb(y):
    emb_func = nn.Embedding(11, 1)
    result = emb_func(y)
    #print(result)
    return result
'''


def train(dataloader, n_epochs=100):
    train_dataloader = dataloader

    # Our network
    net = CNN_GRU().to(device)

    # Our loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)


    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=0.01)

    # Keeping a record of the losses for later viewing
    losses = []
    losses_show = []
    # The training loop
    for epoch in range(n_epochs):
        for x, y in tqdm(train_dataloader):
            # Get some data and prepare the corrupted version
            #print('x测试=',x.shape)
            #x = x.unsqueeze(dim=2)
            #print(x.shape,y.shape)
            x = x.to(device) # Data on the GPU (mapped to (-1, 1))
            #y = class_emb(y)
            #print('y=',y)
            y = y.to(device)
            y = y.squeeze(1)  # 从 (8,1) -> (8,)

            # timestamps = torch.randint(0, 999, (x.shape[0],)).long().to(device)

            # Get the model prediction
            pred = net(x)  # Note that we pass in the labels y

            # Calculate the loss
            #print('pred=',pred.shape,'y=',y.shape)

            loss = loss_fn(pred, y)  # How close is the output to the noise
            #print('loss=',loss)
            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Store the loss for later
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        losses_show.append(avg_loss)
        #print(losses)
        print(f"Finished epoch {epoch}. Average loss values: {avg_loss:05f}")

    print("Saving model...")
    losses_show2 = np.array(losses_show)
    losses_show2 = losses_show2*100
    #print(losses_show2)
    torch.save(net, "classed_mnist_ddpm.pth")
    p = np.linspace(1, n_epochs, n_epochs)


    #print(p)
    plt.plot(p, losses_show2)
    plt.title("迭代次数对模型精确度影响")
    plt.xlabel('迭代次数（次）')
    plt.ylabel('训练损失值（%）')
    plt.show()

    return net


'''
def infer(net):
    # Sampling some different digits:
    # Prepare random x to start from, plus some desired labels y
    x = torch.randn(80, 1, 28, 28).to(device)
    y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)

    # noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    noise_scheduler.set_timesteps(num_inference_steps=40)

    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Get model pred
        with torch.no_grad():
            residual = net(x, t, y)  # Again, note that we pass in our labels y

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

    # Show the results
    x = x.reshape(10, 8, 28, 28).permute(0, 2, 1, 3).reshape(10*28, 8*28)
    img = x.cpu().clip(0, 1).numpy()
    img = (img * 256).astype(np.uint8)
    cv2.imshow("", img)
    cv2.waitKey()
'''


# 读取数据文件夹
def getdata(path):
    data_total = []
    label_total = []
    files = os.listdir(path.encode('utf-8').decode('utf-8'))
    b = 0
    number1 = 0
    for file in files:
        txt_path = path + '/' + file
        paths = os.listdir(txt_path.encode('utf-8').decode('utf-8'))
        print(paths)
        for file_path in paths:
            new_path = txt_path + '/' + file_path
            name = file_path + '=' + str(b)
            #print(name)

            f = open(new_path, 'r', encoding='utf-8')
            line = f.readline()
            data_list = []
            while line:
                num = list(map(str, line.split()))
                data_list.append(num)
                line = f.readline()
            f.close()
            data_array = (np.array(data_list))
            # print(data_array)
            # print(data_array.shape)
            data_array_refined = []
            num = len(data_array)
            #print(num)
            for i in range(num):
                for j in range(19):
                    num1 = 0
                    num2 = 0
                    ten = data_array[i][j][0]
                    if ten == '0':
                        num1 = 0
                    elif ten == '1':
                        num1 = 16
                    elif ten == '2':
                        num1 = 16 * 2
                    elif ten == '3':
                        num1 = 16 * 3
                    elif ten == '4':
                        num1 = 16 * 4
                    elif ten == '5':
                        num1 = 16 * 5
                    elif ten == '6':
                        num1 = 16 * 6
                    elif ten == '7':
                        num1 = 16 * 7
                    elif ten == '8':
                        num1 = 16 * 8
                    elif ten == '9':
                        num1 = 16 * 9
                    elif ten == 'A':
                        num1 = 16 * 10
                    elif ten == 'B':
                        num1 = 16 * 11
                    elif ten == 'C':
                        num1 = 16 * 12
                    elif ten == 'D':
                        num1 = 16 * 13
                    elif ten == 'E':
                        num1 = 16 * 14
                    elif ten == 'F':
                        num1 = 16 * 15
                    else:
                        print('error')
                    single = data_array[i][j][1]
                    if single == '0':
                        num2 = 0
                    elif single == '1':
                        num2 = 1
                    elif single == '2':
                        num2 = 2
                    elif single == '3':
                        num2 = 3
                    elif single == '4':
                        num2 = 4
                    elif single == '5':
                        num2 = 5
                    elif single == '6':
                        num2 = 6
                    elif single == '7':
                        num2 = 7
                    elif single == '8':
                        num2 = 8
                    elif single == '9':
                        num2 = 9
                    elif single == 'A':
                        num2 = 10
                    elif single == 'B':
                        num2 = 11
                    elif single == 'C':
                        num2 = 12
                    elif single == 'D':
                        num2 = 13
                    elif single == 'E':
                        num2 = 14
                    elif single == 'F':
                        num2 = 15
                    else:
                        print('error')
                    total = num1 + num2
                    data_array_refined.append(total)
            data_list_refined = np.array(data_array_refined)
            data_list_refined = data_list_refined.reshape(-1, 19)
            # print(data_list_refined)
            # print(data_list_refined.shape)

            # 将前四行没用的数据删掉
            data_del = np.delete(data_list_refined, [0, 1, 2, 3], axis=1)  # 10863x15
            # print(data_del)

            # 校验是否与校验和一致
            dataset_test = np.delete(data_del, 14, axis=1)
            dataset_sum = np.sum(dataset_test, axis=1)
            dataset_ver = []
            for i in range(len(dataset_sum)):
                s = int(dataset_sum[i]) % 256
                dataset_ver.append(s)
            dataset_final = []
            for i in range(len(data_del)):
                t = data_del[i][14]
                if t == dataset_ver[i]:
                    dataset_final.append(data_del[i])
                else:
                    print('error')

            # 最终数组
            data_list_final = np.array(dataset_final)
            data_list_final = np.delete(data_list_final, [12,13,14], axis=1)
            data_list_final = data_list_final.T
            data_list_final[0] = data_list_final[0] + data_list_final[1] * 256
            data_list_final[2] = data_list_final[2] + data_list_final[3] * 256
            data_list_final[4] = data_list_final[4] + data_list_final[5] * 256
            data_list_final[6] = data_list_final[6] + data_list_final[7] * 256
            data_list_final[8] = data_list_final[8] + data_list_final[9] * 256
            data_list_final[10] = data_list_final[10] + data_list_final[11] * 256
            data_list_final = np.delete(data_list_final, [1, 3, 5, 7, 9, 11], axis=0)

            x = np.array(data_list_final[1])
            y = np.array(data_list_final[3])
            z = np.array(data_list_final[5])
            plt.plot(x)
            plt.plot(y)
            plt.plot(z)
            # plt.plot(peaks, data_list_final[1][peaks], "o")
            plt.xlabel("脉搏波时间序列/2ms", fontsize=18)
            plt.ylabel("脉搏波点值/10Pa", fontsize=18)
            plt.title("脉搏波三合一显示", fontsize=18)
            plt.show()

            data_list_final = np.delete(data_list_final, [0, 2, 4], axis=0)


            #print(data_list_final)
            data_list_final[0] = savgol_filter(data_list_final[0], 5, 3, mode='nearest')
            data_list_final[1] = savgol_filter(data_list_final[1], 5, 3, mode='nearest')
            data_list_final[2] = savgol_filter(data_list_final[2], 5, 3, mode='nearest')

            peaks = scipy.signal.find_peaks_cwt(data_list_final[0], 20)


            #plt.plot(data_list_final[0])
            #plt.plot(peaks, data_list_final[1][peaks], "o")
            #plt.title("脉搏波点位1")

            #plt.show()

            feature_num = 500
            iq = 0
            #label_datas = []
            for i in range(len(peaks)-10):
                iq = iq + 1
                separate_data = []
                for j in range(3):
                    peak_data = data_list_final[j-1][peaks[i+1]:peaks[i + 7]]
                    x = np.arange(0, len(peak_data), 1)
                    y = peak_data
                    z = np.linspace(start=0, stop=len(peak_data)-1, num=feature_num)

                    y_interp = np.interp(z, x, y)
                    new_y = (y_interp-y_interp.min())/(y_interp.max()-y_interp.min()) #插值归一化完成
                    q = np.arange(0, len(y_interp), 1)
                    #plt.plot(q, new_y)
                    #plt.show()
                    separate_data.append(new_y)

                #peaks_array = np.array(separate_data)
                separate_datas = np.array(separate_data)
                separate_datas = separate_datas.T
                #print(separate_datas.shape)

                number1 = number1 + 1
                data_total.append(separate_datas)
            #labels_datas = np.array(label_datas)
            #print(iq)



            # 添加标签
            count = iq
            data_label = np.zeros((count, 1))
            data_label = (data_label + b)
            #print('第' + str(b) + '次')
            label_total.extend(data_label)
            # print(label_total.shape)
        b = b + 1
    data_total_array = np.array(data_total)
    label_total_array = np.array(label_total)
    data_total_array = data_total_array.astype(np.float32)
    label_total_array = label_total_array.astype(np.int64)
    #print('总标签', label_total_array)
    #print(number1)

    return data_total_array, label_total_array




'''
# 单个文件
def get_one_data(file):
    f = pd.read_csv(file, header=None, encoding='utf-8')
    return f
'''


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.08, 1.01 * height, '%s' % float(height), size=8,
                 family="Times new roman")

def main():
    '''
    x = ['RNN', 'GRU', 'LSTM', 'CNN-GRU']
    data = [68.3, 87.2, 84.5, 94.3]
    plt.title("宏平均召回率")
    # plt.grid(ls="--", alpha=0.5)
    plt.xlabel("模型类别")
    plt.ylabel("预测准确率(%)")

    cm = plt.bar(x, data, color=['darkorange','coral','forestgreen','teal'])
    autolabel(cm)
    plt.show()
    '''

    relative_path = '脉象采集'
    data_all, label_all = getdata(relative_path)
    #print(total_all)
    train_dataloader_test = build_dataloader(data_all, label_all, batch_size=8)
    train(train_dataloader_test)
    # train(train_dataloader_test)


if __name__ == "__main__":
    main()
