# Adversarial-Example-Defense-via-Perturbation-Grading-Strategy

代码结构为F&DDefend------NoiseLevel模块
                        |----FLIP+JPEG模块
                        |----DIP模块
                        |----ACC模块


 1.NoiseLevel模块：含有NoiseLevel.m 和 main_NoiseLevel.m,通过调用 main_main_NoiseLevel.m,输入相应的对抗样本路径，可以输出相应的噪声等级。

 2.FLIP+JPEG模块：含有main.m, jpeg.m,fanzhuan.m 和 duqu2.m，通过调用 main.m, 输入QF质量因子，选择是否镜像翻转，可以输出镜像翻转和JPEG压缩后的图像。

 3.DIP模块：含有denoise.py 和 model.py.通过调用 denoise.py，设置参数，可以获取DIP重建之后的图像。

 4.ACC模块：含有yuce.py，通过调用 yuce.py，可以选择不同模型对数据集进行识别，输出识别精度。
