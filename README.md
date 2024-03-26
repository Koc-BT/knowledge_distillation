该项目仅设计学生模型的训练与测试，教师模型的训练移步到 https://github.com/bubbliiiing/unet-pytorch
先将路径修改好，然后运行train.py。
训练完了再运行test.py进行测试。
测试完毕后再运行miou_mpa.py进行计算，te_miou_mpa是生成教师模型计算结果的条形图，其中教师模型的miou与mpa请用教师模型的项目生成。
