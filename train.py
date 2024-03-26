from KDORG import  UNet,train
from nets.unet import Unet


if __name__ == '__main__':
    teacher_model = Unet(2, False)  # 创建教师模型对象
    student_model = UNet(3, 1)  # 创建学生模型对象，输入通道数为3，输出通道数为1
    teacher_model, student_model = train(teacher_model, student_model)  # 调用训练函数，训练教师模型和学生模型


