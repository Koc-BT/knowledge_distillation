from sklearn.metrics import confusion_matrix
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 先运行test.py生成pred文件再运行该程序计算miou与mpa
# 定义一个函数，用于绘制和保存条形图
def plot_and_save_bar_chart(mpa, miou, filename):
    # 创建一个新的图像
    plt.figure()
    # 设置标题
    plt.title('Metrics')
    # 设置x轴标签
    plt.xlabel('type')
    # 设置y轴标签
    plt.ylabel('value')
    # 设置x轴刻度
    plt.xticks([0.25, 0.75], ["MPA", "MIoU"])
    # 设置y轴范围
    plt.ylim(0, 1)
    # 绘制条形图
    bars = plt.bar([0.25, 0.75], [mpa, miou], width=0.4, color=["blue", "green"])
    # 在每个条形上方添加数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')
    # 保存图像到指定文件名
    plt.savefig(filename)
    # 关闭图像
    plt.close()

def compute_miou(conf_mat):
    # 计算IoU
    intersection = np.diag(conf_mat)
    union = np.sum(conf_mat, axis=1) + np.sum(conf_mat, axis=0) - intersection
    IoU = intersection / union
    # 计算mIoU
    mIoU = np.nanmean(IoU)
    return mIoU
def compute_mpa(conf_mat):
    # 计算每个类的准确率
    PA = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    # 计算mPA
    mPA = np.nanmean(PA)
    return mPA



if __name__ == '__main__':

    total_image = 18
    total_mpa = 0
    total_miou = 0  # 初始化IoU总和变量

    for i in range(1, 19):
        # 定义数据预处理
        transform = transforms.Compose([
            transforms.Resize((320, 480)),  # 缩放到统一大小
            transforms.ToTensor(),  # 转换为张量
        ])

        # 读取真实值图片
        mask_test = Image.open(r'D:\KD\ZSZL\img\裂缝\true\%d.png' % i).convert('L')
        mask_test = transform(mask_test)
        mask_test = mask_test.cpu().numpy()
        mask_test = (mask_test > 0).astype(np.uint8)
        # 读取预测值图片
        red_mask = Image.open(r'D:\KD\ZSZL\img_out\pred\%d.png' % i).convert('L')
        red_mask = transform(red_mask)
        red_mask = red_mask.cpu().numpy()
        red_mask = (red_mask > 0).astype(np.uint8)
        #设置类别像素值
        label_to_class = {0: 0, 1: 1}
        pred_labels = np.vectorize(label_to_class.get)(red_mask)
        true_labels = np.vectorize(label_to_class.get)(mask_test)
        # 展平数组以生成一维列表
        pred_labels_flat = pred_labels.flatten()
        true_labels_flat = true_labels.flatten()
        # 计算混淆矩阵
        conf_mat = confusion_matrix(true_labels_flat, pred_labels_flat)
        # 计算mIoU和mPA
        miou = compute_miou(conf_mat)
        mpa = compute_mpa(conf_mat)
        # 累加每个图像的miou
        total_miou += miou
        # 累加每个图像的mpa
        total_mpa += mpa
        # 调用绘制和保存条形图的函数，传入mpa和miou的值，以及想要保存的文件名
        plot_and_save_bar_chart(mpa, miou, r'D:\KD\ZSZL\img_out\指标\%d指标'%i)
        # 打印结果
        print("MPA: {:.4f}".format(mpa))
        print("MIOU: {:.4f}".format(miou))
    # 计算所有图像的平均MIoU
    average_miou = total_miou / total_image
    print("Average MIoU: {:.4f}".format(average_miou))
    # 计算所有图像的平均MPA
    average_mpa = total_mpa / total_image
    print("Average MPa: {:.4f}".format(average_mpa))
    plot_and_save_bar_chart(average_mpa, average_miou, r'D:\KD\ZSZL\img_out\指标\平均指标' )