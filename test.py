# 导入所需的库
import torch
import cv2
from PIL import Image
from KDORG import UNet,test

#测试模式
mode= 'batch_test'

#批量测试
def batch_test():
    for i in range(168,173):
        student_model = UNet(3, 1)
        student_model.load_state_dict(torch.load(r'D:\KD\model_out\student.pth'))
        test_image = Image.open(r'D:\KD\ZSZL\img\裂缝\复杂\%d.jpg' % i)    # 读取测试图片
        output_crack, output_image, red_mask = test(student_model, test_image)   # 调用测试函数，使用学生模型对测试图片进行预测，得到裂缝位置数据和描红后的图片
        cv2.imwrite(r'D:\KD\ZSZL\img_out\%d.png' % i, output_image)  # 保存描红后的图片
        cv2.imwrite(r'D:\KD\ZSZL\img_out\pred\%d.png' % i, red_mask)
#单图测试
def single_test():
    student_model = UNet(3, 1)
    student_model.load_state_dict(torch.load(r'D:\KD\model_out\student.pth'))
    test_image = Image.open(r'D:\KD\ZSZL\img\9.jpg')  # 读取测试图片
    output_crack, output_image , red_mask = test(student_model, test_image)   # 调用测试函数，使用学生模型对测试图片进行预测，得到裂缝位置数据和描红后的图片
    cv2.imwrite(r'D:\KD\ZSZL\img_out\output_image.png', output_image)  # 保存描红后的图片
    cv2.imwrite(r'D:\KD\ZSZL\img_out\red_mask.png', red_mask)


if __name__ == '__main__':
    if mode == 'single_test':
        single_test()
    else:
        batch_test()