# 导入所需的库

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

if __name__ == '__main__':

    mpa = 0.9064
    miou = 0.8193
    # 调用绘制和保存条形图的函数，传入mpa和miou的值，以及想要保存的文件名
    plot_and_save_bar_chart(mpa, miou, r'D:\KD\ZSZL\img_out\指标\TE指标')
    # 打印结果
    print("MPA: {:.4f}".format(mpa))
    print("MIoU: {:.4f}".format(miou))

