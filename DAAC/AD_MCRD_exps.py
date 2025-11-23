"""
AD_comet_mha_expes.py
Function: Run Comet + MHA experiments with different seeds.
History:
20241227    aihongfeng  v1.0
"""
# from comet import COMET
from comet_dual_mha2 import COMET
from models.encoder2 import FTClassifier
# import datautils
from tasks.fine_tuning import finetune_fit
from tasks.fine_tuning import finetune_predict
from tasks.linear_evaluation import eval_classification
from dataloading.ad_preprocessing import load_ad
from config_files.AD_Configs import Config as Configs

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
import copy
import sklearn
from utils import plot_channels
from utils import process_batch_ts
from utils import split_data_label
from utils import start_logging
from utils import stop_logging
from utils import seed_everything

from datetime import datetime
from tqdm import tqdm

configs = Configs()
# RANDOM_SEED = configs.RANDOM_SEED

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# specify saving and logging directory
working_directory = configs.working_directory
dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

logging_directory = configs.logging_directory
if not os.path.exists(logging_directory):
    os.makedirs(logging_directory)

# load and preprocessing data
data_path = "datasets/AD/Feature/"
label_path = "datasets/AD/Label/label.npy"
val_ids = [17,18]  # specify patient ID for validation and test set
test_ids = [19,20]
X_trial_train, X_trial_val, X_trial_test, y_trial_train, y_trial_val, y_trial_test = load_ad(val_ids, test_ids, data_path, label_path)
print(X_trial_train.shape)
print(y_trial_train.shape)
print(X_trial_val.shape)
print(y_trial_val.shape)
print(X_trial_test.shape)
print(y_trial_test.shape)

# normalize data
X_trial_train = process_batch_ts(X_trial_train, normalized=True, bandpass_filter=False)
X_trial_val = process_batch_ts(X_trial_val, normalized=True, bandpass_filter=False)
X_trial_test = process_batch_ts(X_trial_test, normalized=True, bandpass_filter=False)
# print(X_trial_train.shape)
# print(X_trial_val.shape)
# print(X_trial_test.shape)

# Split trail-level data into sample-level data
X_train, y_train = split_data_label(X_trial_train,y_trial_train, sample_timestamps=configs.S_TIMESTAMPS, overlapping=configs.S_OVERLAPPING)
X_val, y_val = split_data_label(X_trial_val,y_trial_val, sample_timestamps=configs.S_TIMESTAMPS, overlapping=configs.S_OVERLAPPING)
X_test, y_test = split_data_label(X_trial_test,y_trial_test, sample_timestamps=configs.S_TIMESTAMPS, overlapping=configs.S_OVERLAPPING)
# print(X_train.shape)
# print(X_val.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)

print('patient:', np.unique(y_test[:,1]))
print('trial:', np.unique(y_test[:,2]))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"The program will run on {device}!")

# @lrq
# 加载新的 AD 数据集
print(' 加载新的 AD 数据集 ....')
data_path0 = "datasets/ADFD/Feature/"
label_path0 = "datasets/ADFD/Label/label.npy"
print(data_path0)
val_ids = [40,41]  # specify patient ID for validation and test set
test_ids = [42,43]
X_trial_train0, X_trial_val0, X_trial_test0, y_trial_train0, y_trial_val0, y_trial_test0 = load_ad(val_ids, test_ids, data_path0, label_path0)
# print(X_trial_train0.shape)
# print(y_trial_train0.shape)
# print(X_trial_val0.shape)
# print(y_trial_val0.shape)
# print(X_trial_test0.shape)
# print(y_trial_test0.shape)


new_order = [6, 7, 2, 3, 0, 1, 10, 11, 8, 9, 4, 5, 12, 13, 14, 15]  # 新的列顺序
# 重新排列列的顺序，保留前 16 列
X_trial_train0 = X_trial_train0[:, :, new_order]
X_trial_val0 = X_trial_val0[:, :, new_order]
X_trial_test0 = X_trial_test0[:, :, new_order]
# print(X_trial_train0_reordered.shape)

# normalize data
X_trial_train0 = process_batch_ts(X_trial_train0, normalized=True, bandpass_filter=False)
X_trial_val0 = process_batch_ts(X_trial_val0, normalized=True, bandpass_filter=False)
X_trial_test0 = process_batch_ts(X_trial_test0, normalized=True, bandpass_filter=False)
print(X_trial_train0.shape)
print(X_trial_val0.shape)
print(X_trial_test0.shape)
# Split trail-level data into sample-level data
X_train0, y_train0 = split_data_label(X_trial_train0,y_trial_train0, sample_timestamps=configs.S_TIMESTAMPS, overlapping=configs.S_OVERLAPPING)
X_val0, y_val0 = split_data_label(X_trial_val0,y_trial_val0, sample_timestamps=configs.S_TIMESTAMPS, overlapping=configs.S_OVERLAPPING)
X_test0, y_test0 = split_data_label(X_trial_test0,y_trial_test0, sample_timestamps=configs.S_TIMESTAMPS, overlapping=configs.S_OVERLAPPING)
print(X_train0.shape)
print(X_val0.shape)
print(X_test0.shape)
print(y_train0.shape)
print(y_val0.shape)
print(y_test0.shape)


# 定义 AE-GAN

# 超参数
input_dim = 16  # 输入数据的维度 (channels)
sequence_length = 256  # 输入数据的序列长度
latent_dim = 64  # 潜在空间维度
output_dim = input_dim  # 输出数据的维度应该与输入相同
batch_size = 32
epochs = 100  # 可以根据需求调整训练轮数
patience=15

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 定义 Encoder 网络结构
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sequence_length):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * sequence_length, latent_dim)  # 将卷积输出展平并映射到潜在空间

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 卷积层 1
        x = torch.relu(self.conv2(x))  # 卷积层 2
        x = x.view(x.size(0), -1)  # 展平 (batch_size, 128 * sequence_length)
        z = self.fc(x)  # 通过全连接层映射到潜在空间
        return z

# 定义 Decoder 网络结构
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, sequence_length):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * sequence_length)  # 将潜在空间向量映射为卷积层的输入
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.fc(z)  # 通过全连接层将潜在空间映射为卷积输入
        x = x.view(x.size(0), 128, sequence_length)  # 变形为 (batch_size, 128, sequence_length)
        x = torch.relu(self.deconv1(x))  # 反卷积层 1
        x = torch.relu(self.deconv2(x))  # 反卷积层 2
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展开为向量
        validity = torch.sigmoid(self.fc1(x))  # 返回 [0, 1] 表示真假
        return validity

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, sequence_length):
        super(Generator, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, sequence_length)
        self.decoder = Decoder(latent_dim, output_dim, sequence_length)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

# AE-GAN 综合模型
class AE_GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(AE_GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        # z = self.encoder(x)
        reconstructed = self.generator(x)
        validity = self.discriminator(reconstructed)
        return reconstructed, validity

# 重建损失 (MSE)
reconstruction_loss_fn = nn.MSELoss()

# 对抗损失 (Binary Cross-Entropy)
adversarial_loss_fn = nn.BCELoss()

# 优化器
lr = 0.02  # 学习率
encoder = Encoder(input_dim, latent_dim, sequence_length)
decoder = Decoder(latent_dim, output_dim,sequence_length)
generator = Generator(latent_dim, output_dim, sequence_length)
discriminator = Discriminator(input_dim)

# 构建模型
model1 = AE_GAN(generator, discriminator).to(device)

# 优化器设置
optimizer_G = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# 重建损失函数
def reconstruction_loss_fn(reconstructed, real_data):
    return nn.MSELoss()(reconstructed, real_data)

# 早停实现：监控验证集损失，若没有改进则停止训练
class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience  # 当验证损失没有改善时，等待的轮数
        self.delta = delta  # 验证损失改进的最小阈值
        self.best_loss = None
        self.best_epoch = 0
        self.counter = 0

    def early_stop(self, validation_loss, epoch):
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss < self.best_loss - self.delta:
            self.best_loss = validation_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            # print(f"Early stopping at epoch {epoch}")
            return True  # 停止训练
        return False

# 训练过程
def train(model, train_loader, val_loader, epochs, optimizer_D, optimizer_G, device, early_stopping):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            real_data = batch[0].float().to(device)  # 输入数据
            # real_labels = batch[1].to(device)  # 标签数据
            batch_size = real_data.size(0)

            # 转置输入数据形状 (batch_size, sequence_length, features) -> (batch_size, features, sequence_length)
            real_data = real_data.permute(0, 2, 1).to(device)

            # 训练判别器
            optimizer_D.zero_grad()

            # 真实数据标签 (1)
            real_labels = torch.ones(batch_size, 1).to(device)
            # 生成数据标签 (0)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # 判别器训练
            reconstructed, validity = model(real_data)
            d_loss_real = adversarial_loss_fn(validity, real_labels)


            # 生成假数据并训练判别器
            fake_data = model.generator(torch.randn(real_data.shape).to(device))
            validity_fake = model.discriminator(fake_data)
            d_loss_fake = adversarial_loss_fn(validity_fake, fake_labels)

            # 计算判别器的损失
            d_loss = (d_loss_real + d_loss_fake)/2
            d_loss.backward()

            # 更新判别器
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()

            # 计算生成器的损失（目标是生成与真实数据相似的数据）
            reconstructed, validity = model(real_data)
            g_loss = adversarial_loss_fn(validity, real_labels) + reconstruction_loss_fn(reconstructed, real_data)
            g_loss.backward()

            # 更新生成器
            optimizer_G.step()

            # 记录损失
            running_loss += d_loss.item() + g_loss.item()

        # 输出当前 epoch 的损失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # 验证集的早停检查
        val_loss = validate(model, val_loader, device)
        if early_stopping.early_stop(val_loss, epoch):
            break  # 如果早停，则停止训练

# 验证过程
def validate(model, val_loader, device):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            real_data = batch[0].float().to(device)
            real_data = real_data.permute(0, 2, 1).to(device)
            reconstructed, _ = model(real_data)
            loss = reconstruction_loss_fn(reconstructed, real_data)
            total_loss += loss.item()
    avg_val_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

# 测试模型的函数
def test_model(model, test_loader, device):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0  # 初始化总损失
    correct = 0
    total = 0

    with torch.no_grad():  # 在测试时，不需要计算梯度
        for batch in test_loader:
            real_data = batch[0].float().to(device)  # 获取测试数据
            # real_labels = batch[1].to(device)  # 获取真实标签
            real_labels = torch.ones(batch_size, 1).to(device)

            # 需要将数据从 (batch_size, seq_len, features) 转换为 (batch_size, features, seq_len)
            real_data = real_data.permute(0, 2, 1).to(device)

            # 预测重建数据以及判别器的输出
            reconstructed, _ = model(real_data)

            # 计算重建损失（这里使用 MSELoss）
            loss = reconstruction_loss_fn(reconstructed, real_data).item()
            total_loss += loss

            # 计算重建误差，比较其是否大于阈值来判断样本是正常还是异常
            reconstruction_errors = torch.mean((real_data - reconstructed) ** 2, dim=(1, 2))
            threshold = 0.1  # 假设的重建误差阈值
            predicted_labels = (reconstruction_errors > threshold).float()
            # print("predicted_labels",predicted_labels.shape)
            # print("real_labels",real_labels.shape)


            # 比较模型预测与真实标签，计算准确率
            correct += (predicted_labels == real_labels).sum().item()
            # print("correct",correct.shape)

            total += real_labels.size(0)
            # print("total",total.shape)

    # 计算平均损失和准确率
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total * 100

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy

# train and test
# 定义你的模型和其他配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AE_GAN(encoder, decoder, discriminator).to(device)  # AE-GAN 模型实例化
optimizer = optim.Adam(model1.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.CrossEntropyLoss()  # 假设是分类任务，你也可以根据需要调整为 MSE

import torch
from torch.utils.data import TensorDataset, DataLoader

# 假设 X_train, X_test, X_val 是您的数据， y_train, y_test, y_val 是对应的标签
# 这里以训练集为例，对数据进行标准化

# 筛选出标签为 1 的数据
train_data_1 = X_train0[y_train0[:,0] == 0]
test_data_1 = X_test0[y_test0[:,0] == 0]
val_data_1 = X_val0[y_val0[:,0] == 0]

# 将 NumPy 数组转换为 PyTorch 张量
train_data_1 = torch.tensor(train_data_1).float()
test_data_1 = torch.tensor(test_data_1).float()
val_data_1 = torch.tensor(val_data_1).float()

# 计算训练集每个特征的均值和标准差
mean = train_data_1.mean(dim=[0, 1], keepdim=True)
std = train_data_1.std(dim=[0, 1], keepdim=True)

# 标准化训练集、测试集和验证集
train_data_1 = (train_data_1 - mean) / std
test_data_1 = (test_data_1 - mean) / std
val_data_1 = (val_data_1 - mean) / std

# 创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(train_data_1)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(test_data_1)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(val_data_1)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

early_stopping = EarlyStopping(patience, delta=0.001)

# 开始训练
train(model1, train_loader, val_loader, epochs, optimizer_D=optimizer_D, optimizer_G=optimizer_G, device=device, early_stopping=early_stopping) # @ lrq epoch
 
# 在测试集上评估模型
test_loss, test_accuracy = test_model(model1, test_loader, device)


# 嵌入新向量

def embed_loss_as_feature_on_data(model, X_train, device, loss_fn):
    model.eval()  # 设置模型为评估模式
    all_data_with_loss = []  # 存储嵌入损失的新数据集

    with torch.no_grad():  # 禁用梯度计算，提高推理效率
        # 确保 X_train 是张量类型
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train)


        # 确保 X_train 是 float 类型并移动到设备上
        X_train = X_train.float().to(device)

        # 获取批次大小
        batch_size = X_train.size(0)

        # 转置输入数据形状 (batch_size, sequence_length, features) -> (batch_size, features, sequence_length)
        real_data = X_train.permute(0, 2, 1)  # 假设输入形状为 (batch_size, sequence_length, features)

        # 使用模型进行推理
        reconstructed, _ = model(real_data)  # 获取重建的输出

        # 计算重建损失
        # 使用 reduction='none' 以获取每个元素的损失，然后计算每个样本的平均损失
        loss_fn_reduction_none = torch.nn.MSELoss(reduction='none')
        reconstruction_loss = loss_fn_reduction_none(reconstructed, real_data)
        # 计算每个样本的平均损失
        reconstruction_loss = reconstruction_loss.mean(dim=[1, 2])  # 形状为 (batch_size,)

        # 将损失作为新特征嵌入到原始数据中
        # 将损失扩展为与输入数据相同的长度
        loss_as_feature = reconstruction_loss.view(-1, 1, 1).expand(batch_size, 1, real_data.size(2))

        # 拼接损失特征到原始数据
        data_with_loss = torch.cat([real_data, loss_as_feature], dim=1)  # 在特征维度拼接

        all_data_with_loss.append(data_with_loss)  # 将新特征的数据添加到列表中

        # 将所有嵌入损失后的数据拼接成一个大的 tensor
        all_data_with_loss = torch.cat(all_data_with_loss, dim=0)

        # 将数据移动到 CPU，然后转换为 NumPy
        all_data_with_loss = all_data_with_loss.transpose(1,2).cpu().numpy()

    return all_data_with_loss


loss_fn = torch.nn.MSELoss()  # 选择损失函数，这里以 MSE 损失为例

# 获取嵌入了损失的新特征数据集
X_train = embed_loss_as_feature_on_data(model1, X_train, device, loss_fn)
X_test = embed_loss_as_feature_on_data(model1, X_test, device, loss_fn)
X_val = embed_loss_as_feature_on_data(model1, X_val, device, loss_fn)

# 现在 new_data_with_loss 就是一个包含原始数据和损失嵌入特征的新数据集
# print(X_train.shape)  # 查看新的数据集形状
if X_train.shape[2]==17:
    configs.input_dims += 1
    # print(configs.input_dims)
# @lrq




# callback functions
def pretrain_callback(model, loss):
    print('saving the weight'+'*'*20)
    n = model.n_epochs
    metrics_dict = {}
    if n % 1 == 0:
        metrics_dict = eval_classification(model, X_train, y_train[:, 0], X_val, y_val[:, 0], fraction=1)
        print(metrics_dict)
        model.save(f"{working_directory}seed{RANDOM_SEED}_pretrain_model.pt")
    return metrics_dict['F1']

def finetune_callback(model, f1, fraction=1.0):
    n = model.n_epochs
    if model.n_epochs == 1:
        model.finetune_f1 = f1
        torch.save(model.state_dict(), f"{working_directory}seed{RANDOM_SEED}_max_f1_{fraction}_finetune_model.pt")
    # control the saving frequency
    if n % 1 == 0:
        if f1 > model.finetune_f1:
            model.finetune_f1 = f1
            torch.save(model.state_dict(), f"{working_directory}seed{RANDOM_SEED}_max_f1_{fraction}_finetune_model.pt")
    return finetune_callback


for RANDOM_SEED in range(41, 46):
    print('='*50)
    configs.RANDOM_SEED = RANDOM_SEED
    print('SEED:', RANDOM_SEED)
    print('='*50)

    total_start_time = datetime.now()
    print('------Self-Supervised For Encoder-----')
    seed_everything(RANDOM_SEED)
    start_time = datetime.now()
    model = COMET(
        input_dims=configs.input_dims,
        device=device,
        lr=configs.pretrain_lr,
        depth=configs.depth,
        batch_size=configs.pretrain_batch_size,
        output_dims=configs.output_dims,

        # @ahf
        pat_multihead=True, 
        tra_multihead=False,
        num_heads=configs.num_heads, 
        head_dim=configs.head_dim, 
        channel_dim=configs.channel_dim,

        flag_use_multi_gpu=configs.flag_use_multi_gpu,
        after_epoch_callback=pretrain_callback,
    )
    print(configs.n_epochs)
    epoch_loss_list, epoch_f1_list = model.fit(
        X_train,
        y_train,
        shuffle_function = configs.shuffle_function,
        verbose=configs.verbose,
        n_epochs=configs.n_epochs,
        masks = configs.masks,
        factors = configs.factors
    )

    end_time = datetime.now()
    print(f'Duration: {end_time - start_time}')


    print('------PFT (freezed Encoder + Trainable Linear)-----')
    # partial training
    start_time = datetime.now()
    seed_everything(RANDOM_SEED)
    pretrain_model = COMET(

        input_dims=configs.input_dims,
        device=device,
        lr=configs.pretrain_lr,
        depth=configs.depth,
        batch_size=configs.pretrain_batch_size,
        output_dims=configs.output_dims,
        flag_use_multi_gpu=configs.flag_use_multi_gpu,
        after_epoch_callback=pretrain_callback,
        
        # @ahf
        pat_multihead=True, 
        tra_multihead=False,
        num_heads=configs.num_heads, 
        head_dim=configs.head_dim, 
        channel_dim=configs.channel_dim,
    )

    pretrain_model.load(f"{working_directory}seed{RANDOM_SEED}_pretrain_model.pt")

    start_logging(RANDOM_SEED, logging_directory)
    val_metrics_dict = eval_classification(pretrain_model, X_train, y_train[:, 0], X_val, y_val[:, 0])
    print("Linear evaluation for validation set\n",val_metrics_dict)
    test_metrics_dict = eval_classification(pretrain_model, X_train, y_train[:, 0], X_test, y_test[:, 0])
    print("Linear evaluation for test set\n",test_metrics_dict)
    print()
    stop_logging()
    end_time = datetime.now()
    print(f'Duration: {end_time - start_time}')

    # the same view are postives across different patients
    # 设置保存图表的目录
    save_directory = "visualization"
    # 确保目录存在
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 设置图表的文件名
    filename = f"seed{RANDOM_SEED}_self_supervised.png"
    # 完整的保存路径
    save_path = os.path.join(save_directory, filename)
    # 创建图表
    plt.figure(1, figsize=(8, 8))
    plt.subplot(121)
    plt.plot(epoch_loss_list)
    plt.title('MCRD_self_supervised: Loss')
    plt.subplot(122)
    plt.plot(epoch_f1_list)
    plt.title('MCRD_self_supervised: Accuracy')

    self_sup_res = pd.DataFrame({'epoch_loss_list':epoch_loss_list,'epoch_f1_list':epoch_f1_list})
    self_sup_res.to_csv('comet_with_dualmha_selfsuper_3_nohierarach_sepmaxpool_catout.csv', index=False)

    # 保存图表到指定路径
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 关闭图表以释放资源
    plt.close()

