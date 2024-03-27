

#                                       智能硬件实验

##                      				             OpenCV 进阶

元培学院	

2000012272

张广欣 



## **一、视频的处理及人脸识别**

### (1) 人脸识别

1. 读取树莓派的摄像头，完成自己人脸、眼睛的识别

分类器通过Haar特征计算和简单分类器的级连实现对特征的提取

完成对人脸的识别，并实时标记

```python
from picamera2 import Picamera2
import time
import numpy as np
import cv2 as cv

#定义分类器
face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

#实例化摄像头，并初始化参数
cam = Picamera2()
cam.still_configuration.main.size = (800,600)
cam.still_configuration.main.format = 'RGB888'
cam.configure('still')
cam.start()
time.sleep(1)

while(True):
    frame = cam.capture_array('main')
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)#将RGB图像转成灰度图
    faces = face_cascade.detectMultiScale(gray,1.3,5)#检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#将检测到的人脸标记在视图中
    cv.imshow('Video',frame)#显示图片
    if cv.waitKey(1) == ord("q"):
        break
cam.stop()
cv.destroyAllWindows()
```

完成对人眼的识别，并实时标记

```python
from picamera2 import Picamera2
import time
import numpy as np
import cv2 as cv

#定义分类器
eye_cascade = cv.CascadeClassifier('./haarcascade_eye.xml')

#实例化摄像头，并初始化参数
cam = Picamera2()
cam.still_configuration.main.size = (800,600)
cam.still_configuration.main.format = 'RGB888'
cam.configure('still')
cam.start()
time.sleep(1)
while(True):#死循环确保程序运行
    frame = cam.capture_array('main')
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)#将RGB图像转成灰度图
    eyes = eye_cascade.detectMultiScale(gray,1.3,5)#检测出图片中所有的眼睛，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
    for (x,y,w,h) in eyes:
        frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#将检测到的眼睛标记在视图中
        
    cv.imshow('Video',frame)#显示图片
    if cv.waitKey(1) == ord("q"):
        break
cam.stop()
cv.destroyAllWindows()
```

2. 完成是否佩戴口罩的简单识别

不依赖识别模型，主要原理可以根据眼睛的位置定位到口罩区域，并判别该区域是否存在口罩来确定该人是否佩戴口罩。

```python
from picamera2 import Picamera2
import time
import numpy as np
import cv2 as cv

#初始化摄像头
eye_cascade = cv.CascadeClassifier('./haarcascade_eye.xml')
cam = Picamera2()
cam.still_configuration.main.size = (800,600)
cam.still_configuration.main.format = 'RGB888'
cam.configure('still')
cam.start()
time.sleep(1)

while(True):
    frame = cam.capture_array('main')
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray,1.3,5)
    #至此，完成对眼睛的识别
    
    if len(eyes) == 2:#为了防止错误识别/只识别到一个眼睛，仅在同时检测到2只眼睛的时候进行以下操作判断
        #口罩区域的起点坐标的确定
        mask_x_begin = min(eyes[0][0],eyes[1][0])
        mask_y_begin = eyes[0][1]+eyes[0][3]
        #口罩尺寸的确定（以眼睛尺寸做参考）
        w1 = int((eyes[0][2]+eyes[1][2])*1.5)
        h1 = int((eyes[0][3]+eyes[1][3])*1.3)
        mask_x_end = mask_x_begin + w1
        mask_y_end = mask_y_begin + h1
        #至此，完成对口罩范围的圈定，以下提取皮肤的颜色
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        skin = cv.inRange(hsv,np.array([0,25,0]),np.array([50,255,255]))
        #过滤出含皮肤颜色的信息，但不包含口罩的颜色信息（默认口罩不是和皮肤颜色一样）
        mask = np.zeros((frame.shape[0], frame.shape[1]),dtype = np.uint8)
        mask[mask_y_begin:mask_y_end, mask_x_begin:mask_x_end] =1#制作关于口罩区域的mask
        out = cv.bitwise_and(skin,skin,mask = mask)#用按照位与的方式，提取口罩区域的颜色图片信息
        mask = np.where((out == 0),0,1).astype('uint8')
        mask_area = mask.sum()#带了口罩的面积
        if mask_area / ((mask_x_end - mask_x_begin)*(mask_y_end - mask_y_begin))>0.8:#如果皮肤颜色占比超过80%，认为，没有带口罩
            print('Please wear your mask.')
        else:
            print('Pass')#带了口罩
        cv.imshow('Video',frame)
    if cv.waitKey(1) == ord("q"):
        break
cam.stop()
cv.destroyAllWindows()
```

### (2)物体跟踪

1. 跟踪移动的乒乓球

使用 BackgroundSubtractor 识别移动的乒乓球，并用方框将其圈出来。使用 contourArea() 判
断边框的大小，过滤一部分噪音。

此算法对提供的视频效果不佳，视频的摄像头一直在移动，全局都有持续晃动被误判断

```python
import cv2 as cv

video = cv.VideoCapture(r'pingpong.mp4')#导入视频
bs  = cv.createBackgroundSubtractorKNN(detectShadows = True)#创建KNN背景减法器
while open:
    ret, frame = video.read()#ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame是帧的图片信息
    if ret == True:
        fgmask = bs.apply(frame)#将背景减法器应用于当前帧，去除背景
        th = cv.threshold(fgmask.copy(),244,255,
                          cv.THRESH_BINARY)[1]#设置阈值，判断图片中的运动物体的信息
        dilated = cv.dilate(th,
                            cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)),
                            iterations = 2)#膨胀算法，优化图片的连通性
        contours, hier = cv.findContours(dilated,
                                         cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#将运动物体的轮廓圈出来
        for i in contours:
            if cv.contourArea(i) > 100:#这一步是为了去除噪音，将小块删掉
                (x,y,w,h) = cv.boundingRect(i)
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)#将运动物体的轮廓用方形圈出
        cv.imshow('Video',frame)
    if cv.waitKey(1) == ord("q"):
        break
video.release()
cv.destroyAllWindows()
```

2. 利用 CAMShift 方法来跟踪乒乓球

   此方法追踪效果很好

```python
import cv2 as cv
import numpy as np

video = cv.VideoCapture(r'mixkit-one-on-one-basketball-game-751-medium.mp4')#读取视频
ret, frame = video.read()#读取第一帧
cv.imshow('Original',frame)
x, y, w, h = cv.selectROI("Original",frame,True,False)#在第一帧中圈出感兴趣，需要追踪的区域
track_window = (x, y, w, h)#确立最初的追踪窗
roi = frame[y:y+h, x:x+w]#确立ROI的区域
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)#将图片转化到HSV空间
roihist = cv.calcHist([hsv_roi],[0],None,[180],[0,180])#计算HSV的直方图反射投影
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)#采用函数cv.normalize()对直方图信息进行规范化
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 1 )#设置终止条件，10次迭代或移动至少1 pt

while open:
    ret, frame = video.read()
   
    if ret == True:
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roihist,[0,180],1)#实现反射投影
        rec, track_window = cv.CamShift(dst, track_window, term_crit)#使用Camshift获取新的位置信息
        x,y,w,h = track_window
        for i in rec:#画出变化后的位置
            img2 = cv.rectangle(frame,(x,y),(x+w,y+h),125, 2)
        cv.imshow('img2',img2)
        if cv.waitKey(1) == ord("q"):
            break
    else:
        break
video.release()
cv.destroyAllWindows()
```



### **三、思考题**

1. **直方图反射投影的基本原理是什么， CAMShift 算法的局限性有哪些？**

   参考opencv官方文档：

      反向投影是一种记录给定图像中的像素点如何适应直方图模型像素分布的方式。首先计算某一特征的直方图模型，然后使用模型去寻找图像中存在的特征。反向投影在某一位置的值就是原图对应位置像素值在原图像中的总数目。

   ​	假设我们已经获得一个肤色直方图(Hue-Staturation),旁边的直方图就是模型直方图(代表手掌的肤色色调)，可以通过掩码操作来抓取手掌所在区域的直方图：

   ![](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221028221756090.png)

   ​	

   流程：

   a. 对测试图像中的每个像素 $p(i,j)$,获取色调数据$（h_{i,j},S_{i,j}）$并找到该色调在直方图中的$bin$位置

   b. 查询模型直方图中对应的$bin-（h_{i,j},S_{i,j}）$并读取该$bin$的数值。
   c. 将此数值存储在新的图像中(BackProjection)。也可以先归一化模型直方图，这样测试图像的输出就可以在屏幕上显示了。
   d. 通过对测试中的图像中的每个像素采用以上步骤，可以得到如下的BackProjection结果图：![image-20221028222043425](/Users/zhangguangxin/Library/Application Support/typora-user-images/image-20221028222043425.png)

   e. 使用统计学的语言，BackProjection中存储的数值代表了测试图像中该像素属于皮肤区域的概率。以上图为例，亮的区域是皮肤区域的可能性更大，而暗的区域则表示更低的可能性。

   

   局限性：

   ​	Camshift适用于物体表面颜色较为单一，且和背景颜色差距较大。

   ​	对于复杂背景或者纹理丰富的物体跟踪效果较差。因为Camshift是对直方图反投影所形成的二值图像进行处理的，如果背景较为复杂或者物体的纹理较为丰富，那么此二值图像的噪声就很多（具体原因可参考直方图反投影的原理），这将直接干扰Camshift对物体位置的判断。因为它单纯考虑颜色直方图，忽略了目标的空间分布特性，这种情况下可以加入对跟踪目标的预测算法。

   

   

   

   

   

   

   

   

   

   

   

   

   

   

   

   

   

   

   

   

   

   

