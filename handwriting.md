

#                                       智能硬件实验

##                      				             手写数字识别

元培学院	

2000012272

张广欣 



### 一、实验目的

1. 了解 SPI 总线传输的原理。
2. 了解 OLED 设备基本原理。
3. 掌握支持向量机的基本原理。

### 二、实验内容

**1.OLED 设备测试**

(1) 通过 SPI 设备访问 OLED，在显示屏上显示 “hello world!”。

```python
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import time
import spidev as SPI
import SSD1306 #这个文件是预先提供的接口函数

#各个端口设置
RST = 19
DC = 16
bus = 0
device = 0

#规定通信方法
disp = SSD1306.SSD1306(rst = RST, dc = DC, spi = SPI.SpiDev(bus,device))
disp.begin()
disp.clear()
disp.display()

#设置字体
font = ImageFont.load_default()
image = Image.new('1',(disp.width,disp.height),'black' )#新建,单色、尺寸和背景颜色
draw = ImageDraw.Draw(image)
draw.text((2,26),'Hello World!',font = font,fill = 'white')
disp.image(image)
disp.display()
```

（2）使用 OLED 显示模块显示一幅图片

```python
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import time
import spidev as SPI
import SSD1306#这个文件是预先提供的接口函数

#各个端口设置
RST = 19
DC = 16
bus = 0
device = 0
disp = SSD1306.SSD1306(rst = RST, dc = DC, spi = SPI.SpiDev(bus,device))
disp.begin()
disp.clear()
disp.display()

font = ImageFont.load_default()#设置字体
image = Image.new('1',(disp.width,disp.height),'black' )
pku_size = [disp.height// 4 * 3, disp.height// 4 * 3] 
pku = Image.open('/home/student/pku2.png').resize((disp.height// 4 * 3,disp.height// 4 * 3),Image.ANTIALIAS)#索引图片位置，修改尺寸，选择图片的显示模式
image.paste(pku,(0,16,pku_size[0],pku_size[1] + 16))#第一个参数是图片信息，第二个参数是图片的位置
draw = ImageDraw.Draw(image)
disp.image(image)
disp.display()
```



**2.验证支持向量机示例代码**

```python
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

#加载数据集
digits = datasets.load_digits()
#用zip函数将图片和标签打包
image_and_labels = list(zip(digits.images, digits.target)) 
#展示训练集的标签和图片
for index, (image, label) in enumerate(image_and_labels[:4]):
  plt.subplot(2, 4, index + 1)
  plt.axis('off')
  plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
  plt.title('Training:%i' % label)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) #预处理训练集，拉成1*64向量方便送入神经网络
classifier = svm.SVC(kernel='rbf', gamma=0.001) #规定高斯核，和使用的参数
#分出训练集和测试集，各为一半
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2]) 
#测试集标签
expected = digits.target[n_samples // 2:] 
#在测试集上
predicted = classifier.predict(data[n_samples // 2:]) 
# 打印预测结果
print("CLassification report for classifier %s:\n%s\n" % (classifier,
metrics.classification_report(expected, predicted)))
#输出混淆矩阵
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#数据集和预测打包
image_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
#输出前四个测试集数据的测试结果
for index, (image, prediction) in enumerate(image_and_predictions[:4]):
  plt.subplot(2, 4, index + 5)
  plt.axis('off')
  plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
  plt.title("Prediction:%i" % prediction)
plt.show()
```

（1）使用的不同参数的同种核和默认参数的不同核，结果如下：

**kernel='rbf',gamma=0.001**         **Accuracy = 0.97**

![image-20221015233433050](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221015233433050.png)

![](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221015235716772.png)

**kernel='rbf',C=100**         **Accuracy = 0.97** 同样的核，更换了参数

![](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221016000553510.png)



![image-20221016000511930](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221016000511930.png)

**kernel='poly':**    **degree = 4**   **accuarcy = 0.96**

![image-20221015233555062](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221015233555062.png)

![image-20221015235921072](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221015235921072.png)

**kernel='poly':**    **degree = 6**   **Accuarcy = 0.95** 多项式核，阶数改变效果不同

![image-20221015233555062](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221015233555062.png)

![image-20221016000049566](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221016000049566.png)**kernel='sigmoid':**       **accuarcy = 0.89**

![image-20221016000237492](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221016000237492.png)

![image-20221016000311252](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221016000311252.png)

在使用的核中，sigmoid的分类效果最差（尤其是1，9）

高斯核效果最好

（2）参数含义：

对于混淆矩阵：

![img](https://pic1.zhimg.com/v2-070a498ad566448714ad4186c5ade634_r.jpg)


$$
Precision\triangleq \frac{TP}{TP+FP}
\\
Recall\triangleq \frac{TP}{TP+FN} 
\\
Acurracy\triangleq \frac{TP+TN}{TP+FP+TN+FN} 
\\
F_1 score = 2 * \frac{Precision*Recall}{Precision+Recall}
$$




3. **使用 OLED 显示结果**

```python
import matplotlib.pyplot as plt
from sklearn import datasets , svm, metrics
import time
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import time
import spidev as SPI
import SSD1306
import cv2 as cv
#设置各个端口，以及通信参数
RST = 19
DC = 16
bus = 0
device = 0
disp = SSD1306.SSD1306(rst = RST, dc = DC, spi = SPI.SpiDev(bus,device))
disp.begin()
disp.clear()
disp.display()
font = ImageFont.load_default()
#从上一块代码中摘出需要的部分，加载数据集训练模型
digits = datasets.load_digits() 
images_and_labels = list(zip(digits.images , digits.target))
n_samples = len(digits.images)
data = digits.images.reshape((n_samples , -1))
classifier = svm.SVC(gamma = 0.001)
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
#将测试集的图片和预测结果显示在OLED上
for index , (image , prediction) in enumerate(images_and_predictions [:10]):
    digit = Image.fromarray((image*8).astype(np.uint8),mode = 'L').resize((48,48)).convert('1')
    img = Image.new('1',(disp.width,disp.height),'black' )
    img.paste(digit,(0,16,digit.size[0],digit.size[1] + 16))#图片
    draw = ImageDraw.Draw(img)
    result = 'predicted:'+ str(prediction)
    draw.text((50,26),result,font = font,fill = 'white')#文字
    disp.image(img)
    disp.display()
    input()#用此办法实现按“Enter”键切换
```

4. **使用摄像头读入手写数字**

   ```python
   from picamera2 import Picamera2
   import time
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn import datasets , svm, metrics
   import time
   import numpy as np
   import cv2
   #首先拍照得到图片并储存
   cam = Picamera2()
   cam.still_configuration.main.size = (800,600)
   cam.still_configuration.main.format = 'RGB888'
   cam.configure('still')
   cam.start()
   time.sleep(1)
   while(True):
       frame = cam.capture_array('main')
       cv2.imshow('Video Test',frame)
       if cv2.waitKey(1) == ord("q"):
           cv2.imwrite('handwriting9.tif',frame)#比如：数字9
           print('Photo get!')
           break
   cam.stop()
   cv2.destroyAllWindows()
   #读取图片
   img = cv.imread('handwriting9.tif',1)
   img1 = img[50:550,200:600]#截取数字
   img2 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)#转成灰度图
   img2 = cv.resize(img2,(8,8),interpolation = cv.INTER_CUBIC)#调整尺寸至8*8，和训练数据集相同
   img2 = 16 - img2/16#由于训练集是黑底白字，也做此翻转
   digits = datasets.load_digits() 
   images_and_labels = list(zip(digits.images , digits.target))
   n_samples = len(digits.images)
   data = digits.images.reshape((n_samples , -1))
   classifier = svm.SVC(gamma = 0.001)            
   classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])#训练模型
   hand_data = img2.reshape((1 , -1))#将手写图片拉成一维向量
   predicted = classifier.predict(hand_data)#神经网络预测手写图片
   print(predicted)
   ```
   

### **三、思考题**

1. **为什么不把全部的数据都用来做训练呢？然后在训练数据集上进行验证效果一定会更好！**

   ​	神经网络如果将全部数据作为训练集，且使用训练集进行验证，由于是在已经训练过的数据集上进行验证，没有办法获知神经网络的泛化能力，会造成模型在训练集上的过拟合。

   

2. **将具有 16 级灰度的图像显示在单色屏上，能保证显示效果的原理是什么？**

   ​    led显示屏灰度也就是所谓的色阶或灰阶，是指亮度的明暗程度。对于数字化的显示技术而言，灰度是显示色彩数的决定因素。一般而言灰度越高，显示屏画面显示的色彩越丰富，画面也越细腻，更易表现丰富的细节。

   控制LED灰度的方法：

   1. 改变流过LED的电流。一般LED管允许连续工作电流在20毫安左右，除了红色LED有饱和现象外，其他LED灰度基本上与流过的电流成比例
   2. 利用人眼的视觉惰性，用脉宽调制方法来实现灰度控制，也就是周期性改变光脉冲宽度(即占空比)，只要这个重复点亮的周期足够短(即刷新频率足够高)，人眼是感觉不到发光像素在抖动。
   3. 在一定面积的像素点内电量灯管的多少也可以让人眼产生灰度的视觉效果。