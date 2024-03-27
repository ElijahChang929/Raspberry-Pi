

#                                       智能硬件实验

##                      				             人工神经网络

元培学院	

2000012272

张广欣 

### 一、实验目的

1. 熟悉 AD/DA 原理，实现控制发光二极管开、关、调亮、调暗的功能。
2. 学习通过 PyTorch 搭建人工神经网络模型的方法。
3. 学习通过卷积神经网络（CNN）实现简单手势识别的方法，掌握调整 CNN 网络参数的方法
4. 实现通过手势识别实时控制发光二极管亮度的功能。
   

### 二、实验内容

**1. AD/DA模块**

用电位器控制发光二极管的明暗

```python
import smbus
import time
#声明发光二极管和电位器的地址
address  = 0x48
A0 = 0x40
bus = smbus.SMBus(1)
while True:#死循环保证程序的运行
  #实现的逻辑：虽然电位器和发光二极管并没有直接并联，但可以通过调节电位器并用树莓派读取该引脚的数值来实现可变的模拟-数字输入。通过另一端口将此输入输出到LED灯上，实现间接控制
    value = bus.read_byte(address)#读取电位器的数值
    bus.write_byte_data(address,A0,value)#控制LED灯
```

**2.通过 OpenCV 获取训练样本并建立训练集和测试集**

测试该网络的性能，得到loss随epoch变化的曲线（示例代码）

```python
# 此代码用于训练及测试神经网络
# 需要在此代码所在的目录下存放一个数据集文件夹
# 数据集文件夹包含“train”、“test”两个子文件夹
# 两个子文件夹下均有四个以手势类型命名的文件夹，存放着样本
# 训练部分使用小批量梯度下降、Adam优化器
# 使用tensorboard的SummaryWriter记录每轮训练后在训练集、测试集上的损失、准确率等情况
# SummaryWriter、训练好的模型将保存在数据集文件夹下
# 测试部分将给出模型在测试集上的混淆矩阵
# 混淆矩阵图片将保存在数据集文件夹下

DATA_FOLDER = 'dataset'  # 数据集文件夹名
DO_TRAIN = True  # 是否进行训练，True需要有数据集
DO_TEST = True  # 是否进行测试，True需要DO_TRAIN为True或数据集文件夹下已存在一个网络模型
SAVE_MODEL = True  # 是否保存训练好的模型，True需要DO_TRAIN为True
SAVE_CMFIG = True  # 是否保存混淆矩阵，True需要DO_TEST为True
LEARNING_RATE = 0.001  # Adam优化器学习率
MAX_EPOCH = 10  # 进行epoch数
BATCH_SIZE = 10  # 每批含样本数

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import itertools
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

writer = SummaryWriter()  # 用于记录训练
torch.set_grad_enabled(True)

train_set = torchvision.datasets.ImageFolder(f'./{DATA_FOLDER}/train', transform=transforms.Compose(
    [transforms.Grayscale(1), transforms.ToTensor()]))  # 从数据集文件夹导入训练样本，灰度化并转为张量
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True)  # 训练集加载器，随机打乱训练集并以BATCH_SIZE个为一批


# 神经网络
# 采用“输入-卷积-池化-卷积-池化-一维化-全连接-输出”的结构
# 激活函数为ReLu
# 图片原始大小为80*60
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=12 * 3 * 2, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=30)
        self.out = nn.Linear(in_features=30, out_features=4)

    def forward(self, t):
        t = F.relu(self.conv1(t))  # -> 76*56
        t = F.max_pool2d(t, kernel_size=4, stride=4)  # ->19*14
        t = F.relu(self.conv2(t))  # -> 15*10
        t = F.max_pool2d(t, kernel_size=4, stride=4)  # ->3*2
        t = t.reshape(-1, 12 * 3 * 2)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        return self.out(t)


# 训练部分
def train():
    network = Network()
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    images, labels = next(iter(train_loader))  # 从前面的加载器中按批获得样本

    # 在tensorboard中记录一批样本的图像，添加网络，以监视样本或网络可能的异常
    grid = torchvision.utils.make_grid(images)  # 将一批样本做成网格
    tb = SummaryWriter(f'./{DATA_FOLDER}')  # SummaryWriter将保存在数据集文件夹下
    tb.add_image('images', grid)  # 添加样本网格
    #tb.add_graph(network, images)  # 添加网络  

    # 比较网络对一批样本的预测preds与样本的真实标签labels，返回预测正确的个数
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    # 训练MAX_EPOCH轮
    for epoch in range(MAX_EPOCH):
        t0 = time.time()  # 记录用时
        total_loss = 0  # 记录训练集上的损失
        total_correct = 0  # 记录训练集上的正确个数
        for batch in train_loader:  # 按批进行
            images, labels = batch
            preds = network(images)  # 获得目前网络对这一批的预测
            loss = F.cross_entropy(preds, labels)  # 计算交叉熵
            optimizer.zero_grad()  # 梯度清零，避免循环时backward()累加梯度
            loss.backward()  # 反向传播求解梯度
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 更新训练集上的损失
            total_correct += get_num_correct(preds, labels)  # 更新训练集上正确个数
        # 一次epoch完成，开始在测试集上看效果
        pred_set = torchvision.datasets.ImageFolder(f'./{DATA_FOLDER}/test', transform=transforms.Compose(
            [transforms.Grayscale(1), transforms.ToTensor()]))  # 按和前面train_set一样的方法获得测试集pred_set
        prediction_loader = torch.utils.data.DataLoader(pred_set, batch_size=BATCH_SIZE)  # 以及对应加载器
        p_total_loss = 0  # 记录在训练集上的损失
        p_total_correct = 0  # 记录训练集上总正确数
        for batch in prediction_loader:
            images, labels = batch
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            p_total_loss += loss.item()
            p_total_correct += get_num_correct(preds, labels)

        # 在终端显示epoch数、准确率、用时，监视训练进程
        print('epoch', epoch + 1, 'total_correct:',
              total_correct, 'loss:', total_loss)
        print('train_set accuracy:', total_correct / len(train_set))
        print('test_set accuracy:', p_total_correct / len(pred_set))
        print('time spent:', time.time() - t0)

        # 在SummaryWriter中分别记录训练集和测试集的损失、准确率、正确数信息
        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Prediction Loss', p_total_loss, epoch)
        tb.add_scalar('Number Correct', total_correct, epoch)
        tb.add_scalar('Prediction Number Correct', p_total_correct, epoch)
        tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)
        tb.add_scalar('Prediction Accuracy', p_total_correct / len(pred_set), epoch)
    tb.close()
    if SAVE_MODEL:  # 保存模型在数据集文件夹下
        torch.save(network, F'./{DATA_FOLDER}/network.pkl')


if DO_TRAIN:
    train()


# 测试模型，绘制混淆矩阵
def test(network):
    def get_all_preds(model, loader):
        # 传入网络模型和加载器，获得模型对加载器中样本的全部预测
        all_preds = torch.tensor([])
        for batch in loader:
            images, labels = batch
            preds = model(images)  # 按批获得预测
            all_preds = torch.cat((all_preds, preds), dim=0)  # 再拼接起来
        return all_preds

    with torch.no_grad():  # 测试时不需要计算梯度等数据，节省资源
        pred_set = torchvision.datasets.ImageFolder(f'./{DATA_FOLDER}/test', transform=transforms.Compose(
            [transforms.Grayscale(1), transforms.ToTensor()]))  # 导入测试集
        prediction_loader = torch.utils.data.DataLoader(pred_set, batch_size=BATCH_SIZE)  # 对应加载器，由于是测试，无需shuffle
        train_preds = get_all_preds(network, prediction_loader)  # 获取网络对测试集数据的预测
    total_types = len(pred_set.classes)  # 总分类数
    stacked = torch.stack((torch.tensor(pred_set.targets), train_preds.argmax(dim=1)), dim=1)  # 拼接测试样本的真实类型与预测类型
    cmt = torch.zeros(total_types, total_types, dtype=torch.int32)  # 初始化混淆矩阵
    for p in stacked:  # 根据测试样本的真实类型与预测类型，在混淆矩阵的对应位置计数+1
        train_label, predicted_label = p.tolist()
        cmt[train_label, predicted_label] = cmt[train_label, predicted_label] + 1

    def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
        # 传入ndarray型的混淆矩阵，分类名，标题，配色；绘制上文得到的混淆矩阵
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()  # 设置颜色渐变条
        tick_marks = np.arange(len(classes))  # 图像有分类数个刻度
        plt.xticks(tick_marks, classes, rotation=45)  # 用分类名标签xy刻度
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        # 在表格上对应位置显示数字，为可视化效果，以其中最大数据的一半为界确定文字颜色的黑白
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()  # 填充图像区域
        plt.ylabel('True label')  # 命名坐标
        plt.xlabel('Predicted label')

    # 绘图
    plt.figure(figsize=(4, 4))
    plot_confusion_matrix(cmt.numpy(), pred_set.classes)
    if SAVE_CMFIG:  # 将混淆矩阵图片保存在数据集文件夹下
        plt.savefig(f'./{DATA_FOLDER}/confusion_matrix.png')
    plt.show()


if DO_TEST:
    test(torch.load(f'./{DATA_FOLDER}/network.pkl'))
    
    
    
```

​	使用Tensorboard得到Loss随epoch的变化曲线：

![image-20221106210655931](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221106210655931.png)

​	得到的混淆矩阵：

![image-20221106211030110](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221106211030110.png)



**2.根据所提供的代码，捕捉自己的手势建立测试集**

使用摄像头，建立自己的数据集

```python
FOLDER_NAME = './own_dataset/test/Right' # train , test / Five , Palm , Left , Right
START_INDEX = 1
CAPTURE_CNT = 1000
from picamera2 import Picamera2
import numpy as np
import cv2
import time

cam = Picamera2()
cam.still_configuration.main.size = (800,600)
cam.still_configuration.main.format = 'RGB888'
cam.configure('still')
cam.start()
time.sleep(1)
# 开 始 捕 捉 前 对 效 果 进 行 预 览

while True:
    frame = cam.capture_array('main')
    HSV = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV) # HSV颜 色 空 间 的 图 像
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) # 灰 度 化 的 图 像
    image_mask = cv2.inRange(HSV, np.array([0, 4, 0]), np.array([12, 255, 255])) # 实 验 室 相 机 的 肤 色
    output = cv2.bitwise_and(gray , gray , mask=image_mask) # 按 颜 色 空 间 提 取 手80 第八章 人工神经网络
    output = cv2.resize(output , (80, 60)) # 缩 小 图 片
    output = cv2.blur(output , (2, 2)) # 模 糊 处 理
    cv2.imshow('orig', frame) # 显 示 预 览
    cv2.imshow('gray', output)
    if cv2.waitKey(1) == ord('q'): # 按q退 出 预 览
        break

 # 开 始 捕 捉 样 本
index = START_INDEX # 样 本 命 名 的 编 号， 从 START_INDEX 以1为 步 距 往 上 递 增
while index < START_INDEX + CAPTURE_CNT: # 取 样 CAPTURE_CNT 个
    print(f'process: {index}/{CAPTURE_CNT}')
    frame = cam.capture_array('main')
    HSV = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    # image_mask = cv2.inRange(HSV, np.array([0, 48, 80]), np.array([20, 255,255])) # 自 己 电 脑 相 机
    image_mask = cv2.inRange(HSV, np.array([0, 4, 0]), np.array([12, 255, 255])) # 实 验 室 相 机
    output = cv2.bitwise_and(gray , gray , mask=image_mask)
    output = cv2.resize(output , (80, 60))
    output = cv2.blur(output , (2, 2))
    cv2.imshow('orig', frame)
    cv2.imshow('gray', output)
    cv2.imwrite(f'{FOLDER_NAME}/{index}.png', output) # 保 存 到 FOLDER_NAME 下
    index += 1
    time.sleep(0.1) # 取 样 间 隔
    if cv2.waitKey(1) == ord('w'): # 可 按w中 断
         break
        
cv2.destroyAllWindows()
cam.release()
```

使用此数据集进行训练，得到模型的混淆矩阵：

![image-20221106214128540](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221106214128540.png)

**3.通过手势识别实时控制发光二极管的亮度**

```python
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from picamera2 import Picamera2
import numpy as np
import cv2
import time
import smbus

# 声明二极管的地址
address  = 0x48
A0 = 0x40
bus = smbus.SMBus(1)

# 神经网络同训练时的网络结构
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=12 * 3 * 2, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=30)
        self.out = nn.Linear(in_features=30, out_features=4)

    def forward(self, t):
        t = F.relu(self.conv1(t))  # -> 76*56
        t = F.max_pool2d(t, kernel_size=4, stride=4)  # ->19*14
        t = F.relu(self.conv2(t))  # -> 15*10
        t = F.max_pool2d(t, kernel_size=4, stride=4)  # ->3*2
        t = t.reshape(-1, 12 * 3 * 2)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        return self.out(t)

#load 网络训练的参数
network = torch.load('/home/student/zgx/lab7/dataset/network.pkl')

#初始化摄像机
cam = Picamera2()
cam.still_configuration.main.size = (800,600)
cam.still_configuration.main.format = 'RGB888'
cam.configure('still')
cam.start()
time.sleep(1)

#flag 用于表示灯的开关状态
#light是初始亮度值

flag = True
light = 250

#Right 3
#Palm 2
#Left 1
#five 0

def LED(x):
    global flag
    global light
    # 以下是输入以及对应的合法状态产生的结果，如果不是以下状态，则不产生反馈，命令者需要调整命令
    if x == 1  and flag == True and light <= 250:#当灯为开，没有达到最大亮度时，可以调亮灯
        light += 5
        print('Brighter')
        bus.write_byte_data(address,A0,light)
        return
    elif x == 0 and flag == False:#当灯为关时，可以开灯
        bus.write_byte_data(address,A0,light)#打开时，亮度与上一次关灯的亮度相同
        print('On')
        flag = True
        return
    elif x == 2 and flag == True:#当灯为开时，可以关灯
        bus.write_byte_data(address,A0,0)
        print('Off')
        flag = False
        return
    elif x == 3 and flag == True and light > 0:#当灯为关，亮度没有到0时，可以调暗灯
        light -= 5
        print('Darker')
        bus.write_byte_data(address,A0,light)
        return
#识别手势，做出反馈
while True:
    frame = cam.capture_array('main')
    HSV = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    image_mask = cv2.inRange(HSV, np.array([0, 4, 0]), np.array([12, 255, 255])) #调整了HSV空间参数
    output = cv2.bitwise_and(gray , gray , mask=image_mask)
    output = cv2.resize(output , (80, 60))
    output = cv2.blur(output , (2, 2))
    #通过一系列处理，把摄像头得到的图片处理成与训练集相同的状态
    data = torch.tensor([[output]], dtype=torch.float) # 将获取的实时样本转为可传入网络的tensor
    pred_scores = network(data) # 获取各类型的分数
    prediction = pred_scores.argmax(dim=1).item() # 取最大者为结果
    cv2.imshow('gesture',output)
    LED(prediction)#用预测结果控制灯
    time.sleep(0.1)# 防止短时间连续触发
    if cv2.waitKey(1) == ord('w'): # 可 按w中 断
         break
cam.stop()
cv2.destroyAllWindows()
```

增加了一个卷积+池化层后，对于手势识别的正确率有所提升

### **三、思考题**

1.如何改进容易出现误判或得分较低的姿势的正确率？

​	 出现这样的问题，本质原因应该为训练集中该手势的样本数不够多/样本的多样性不够。因此可以：

（1）增大该姿势的样本数

（2）改进样本质量，优化训练集：让训练集图片的状态和实际测试的情况尽可能相似，同时该姿势各个角度，各个方位，各种远近的图片都要加到训练集中

当然，还可以通过调整网络结构来整体增强网络的识别能力。