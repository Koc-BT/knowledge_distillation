# 导入所需的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


# 定义UNET网络结构
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # 编码器部分
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 解码器部分
        self.upconv4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv4 = nn.Conv2d(384, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.upconv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv5 = nn.Conv2d(192, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.upconv6 = nn.ConvTranspose2d(64, out_channels, 2, stride=2)

    def forward(self, x):
        # 编码器部分
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x1)
        x2 = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x2)
        x3 = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool3(x3)
        # 解码器部分
        x = torch.cat([x3, self.upconv4(x)], dim=1)  # 跳跃连接
        x = self.relu4(self.bn4(self.conv4(x)))
        x = torch.cat([x2, self.upconv5(x)], dim=1)  # 跳跃连接
        x = self.relu5(self.bn5(self.conv5(x)))
        x = torch.sigmoid(self.upconv6(x))  # 输出概率图
        return x



def FocalLoss( y_pred_student, y_true):
    at = 0.25
    γ = 2
    # 计算交叉熵损失
    ce_loss = nn.functional.binary_cross_entropy_with_logits(y_pred_student, y_true, reduction='none')
    # 计算预测概率
    pt = torch.exp(-ce_loss)
    # 计算focal loss
    focal_loss = at * (1 - pt) ** γ * ce_loss
    # 返回平均focal loss
    return torch.mean(focal_loss)

# 定义KL损失函数
def KL_loss(y_pred_student, y_pred_teacher):
    T = 10  # 温度参数，可调节
    kl_loss =  F.kl_div(F.log_softmax(y_pred_teacher / T), F.softmax(y_pred_student / T) )  # 计算KL散度损失
    loss = T * T * kl_loss
    return loss


#定义一个二分类的dice_loss损失函数
def dice_loss(y_pred_student, y_true, smooth=1.):
    # 将预测值和目标值转换为一维向量
    y_pred_student = y_pred_student.view(-1)
    y_true = y_true.view(-1)
    # 计算交集和并集
    intersection = (y_pred_student * y_true).sum()
    union = y_pred_student.sum() + y_true.sum()
    # 计算dice系数
    dice = (2. * intersection + smooth) / (union + smooth)
    # 返回dice loss
    return 1 - dice


# 定义数据集类，加载voc数据集中的三通道有裂缝的图片和该图片对应的单通道裂缝灰度图片
class CrackDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 数据集根目录
        self.transform = transform  # 数据预处理方法
        self.image_list = os.listdir(os.path.join(root_dir, r'D:\KD\ZSZL\VOCdevkit\VOC2007\JPEGImages'))  # 图片文件列表

    def __len__(self):
        return len(self.image_list)  # 数据集大小

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.image_list[idx]  # 图片文件名
        image_path = os.path.join(self.root_dir, r'D:\KD\ZSZL\VOCdevkit\VOC2007\JPEGImages', image_name)  # 图片文件路径
        mask_path = os.path.join(self.root_dir, r'D:\KD\ZSZL\VOCdevkit\VOC2007\SegmentationClass', image_name.replace('.jpg', '.png'))  # 掩码文件路径
        self.transform = transforms.Compose([  # 定义数据预处理和增强
            transforms.Resize((320, 480)),  # 缩放到统一大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])
        self.transform_mask = transforms.Compose([  # 定义数据预处理和增强
            transforms.Resize((320, 480)),  # 缩放到统一大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(0.5, 0.5)  # 归一化
        ])
        image = Image.open(image_path)# 用PIL库打开图片
        mask = Image.open(mask_path) # 用PIL库打开掩码

        if self.transform:
            image = self.transform(image)  # 对图片进行预处理
        if self.transform_mask:
            mask = self.transform_mask(mask)  # 对掩码进行预处理

        return image, mask  # 返回图片和掩码


# 定义训练函数，输入教师模型和学生模型，输出训练后的模型
def train(teacher_model, student_model):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 判断是否有GPU可用，如果有则使用GPU，否则使用CPU
    teacher_model.to(device)  # 将教师模型移动到设备上
    student_model.to(device)  # 将学生模型移动到设备上
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.006)  # 定义优化器，只优化学生模型的参数，教师模型的参数保持不变
    #scheduler = CosineAnnealingLR(optimizer, T_max=400, eta_min=0.000001)  # T_max是半个周期的epoch数，eta_min是最小学习率
    dataset = CrackDataset('voc')  # 创建数据集对象，加载voc数据集
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=12)  # 创建数据加载器对象，设置批大小为4，打乱数据，使用12个进程加载数据
    epochs = 600# 设置训练轮数为600，可调节
    loss_list = []  # 定义一个列表用于存储每个epoch的loss值
    # 定义一个标签归一化的函数，将标签值映射到[0,1]的范围内
    def label_normalize(label_matrix):
        # 计算标签矩阵的最大值和最小值
        label_max = torch.max(label_matrix)
        label_min = torch.min(label_matrix)
        # 计算标签矩阵的范围
        label_range = label_max - label_min
        # 使用公式进行归一化
        label_norm = (label_matrix - label_min) / label_range
        # 返回归一化后的标签矩阵
        return label_norm

    for epoch in range(epochs):  # 遍历每一轮训练
        print(f'Epoch {epoch + 1}/{epochs}')  # 打印当前轮数和总轮数
        running_loss = 0.0  # 记录累计损失值
        for i, (images, masks) in enumerate(tqdm(dataloader)):  # 遍历每一个批次的数据，使用tqdm显示进度条
            images = images.to(device)  # 将图片移动到设备上
            masks = masks.to(device)  # 将裂缝灰度图片移动到设备上
            masks = label_normalize(masks)
            optimizer.zero_grad()  # 清空梯度缓存
            y_pred_teacher = teacher_model(images)  # 使用教师模型对图片进行预测，得到裂缝概率图
            y_pred_teacher = label_normalize(y_pred_teacher)
            y_pred_student = student_model(images)  # 使用学生模型对图片进行预测，得到裂缝概率图
            distillation_loss = 0.7* FocalLoss(y_pred_student, masks) + 0.3* KL_loss(y_pred_student, y_pred_teacher)  # 计算知识蒸馏损失函数值
            dice_loss_double = dice_loss(y_pred_student, masks, smooth=1.)  # 计算学生模型与真实标签的dice_loss
            loss =    distillation_loss +  dice_loss_double # 综合损失
            loss.backward()  # 反向传播计算梯度值
            optimizer.step()  # 更新学生模型的参数值
            #scheduler.step()  # 在每个epoch结束后调用step()来更新学习率
            running_loss += loss.item()  # 累加损失值
            if (i + 1) % 10 == 0:  # 每10个批次打印一次平均损失值
                print(f'Batch {i + 1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0  # 重置累计损失值
        print(f'Epoch {epoch + 1} finished, Loss: {running_loss / len(dataloader):.4f}')  # 打印每一轮训练的平均损失值
        loss_list.append(running_loss / len(dataloader))
    # 绘制loss_list中的loss值随epoch变化的曲线图，并保存为png格式的图片
    plt.plot(range(1, epochs + 1), loss_list, label='Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(r'D:\KD\ZSZL\img_out\loss_curve.png')  # 保存图片到当前目录下，文件名为loss_curve.png
    torch.save(teacher_model.state_dict(), r'D:\KD\model_out\teacher.pth')  # 保存教师模型的参数为teacher.pth
    torch.save(student_model.state_dict(), r'D:\KD\model_out\student.pth')  # 保存学生模型的参数为student.pth
    return teacher_model, student_model  # 返回训练后的模型

# 定义测试函数，输入模型和图片，输出裂缝位置数据和描红后的图片
def test(model, image):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 判断是否有GPU可用，如果有则使用GPU，否则使用CPU
    model.to(device)  # 将模型移动到设备上
    model.eval()  # 设置模型为评估模式，不使用dropout和batchnorm
    transform = transforms.Compose([  # 定义数据预处理
        transforms.Resize((320, 480)),  # 缩放到统一大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # 归一化
    ])
    image = transform(image)  # 对图片进行预处理
    image = image.unsqueeze(0)  # 增加一个批次维度
    image = image.to(device)  # 将图片移动到设备上
    y_pred = model(image)  # 使用模型对图片进行预测，得到裂缝概率图
    y_pred = y_pred.squeeze(0)  # 去掉批次维度
    y_pred = y_pred.cpu().detach().numpy()  # 将张量转换为numpy数组
    y_pred = (y_pred > 0.5).astype(np.uint8)  # 将概率图转换为二值图，阈值为0.5，大于0.5的为1，小于等于0.5的为0
    crack_data = y_pred * 255  # 将二值图转换为灰度图，1对应255，表示裂缝位置数据
    image = image.squeeze(0)  # 去掉批次维度
    image = image.cpu().detach().numpy()  # 将张量转换为numpy数组
    image = (image * 0.5 + 0.5) * 255  # 反归一化并转换为灰度值范围
    image = image.transpose(1, 2, 0)  # 调整通道顺序，从(C, H, W)变为(H, W, C)
    image = image.astype(np.uint8)  # 转换为无符号整数类型
    red_mask = np.zeros_like(image)  # 创建一个和图片大小相同的全零数组，用于存放红色遮罩
    red_mask[:, :, 2] = crack_data  # 将裂缝位置数据赋值给红色通道，表示红色遮罩
    alpha = 0.6  # 设置透明度参数，可调节
    output_image = cv2.addWeighted(image, 1 - alpha, red_mask, alpha, 0)  # 将图片和红色遮罩按照透明度叠加，得到描红后的图片
    return crack_data, output_image , red_mask # 返回裂缝位置数据和描红后的图片




