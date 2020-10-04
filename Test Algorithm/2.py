import cv2
import math
import imutils
import numpy as np
import argparse
import time
parse=argparse.ArgumentParser()
parse.add_argument('-v','--video',help='path to the video file')
def roi_mouse(event,x,y,flags,param):#调用鼠标时间进行人工标注
    global x_i,y_i,xo,yo
    if event==cv2.EVENT_LBUTTONDOWN:
        x_i,y_i=x,y
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_LBUTTONDOWN:
        xo,yo=x,y
def sigma(pixel,i):
    if pixel>i*16 and pixel<16*(i+1):
        return 1
    else:
        return 0
args=vars(parse.parse_args())
video=args['video']
if video:
    cap=cv2.VideoCapture(video)
    print(1)
else:
    cap=cv2.VideoCapture(0)
    print(0)
ret,img=cap.read()#读取帧
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换灰度
for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        img[i][j]=int(img[i][j]/20)*20#阈值分割
cv2.namedWindow('image')
cv2.setMouseCallback('image',roi_mouse)#人工标注
cv2.imshow('image',img)
cv2.waitKey()
start=time.time()#开始计时
a=xo-x_i #roi长
b=yo-y_i #roi宽
weight=np.zeros((a,b))
n=16 #n为直方图像素区间总数
q=np.zeros(n)#概率密度数组
sum=np.zeros(n)
h=math.sqrt(a*a+b*b)
C=0
fx=x_i+a/2
x0=fx
fy=y_i+b/2
y0=fy
for xi in range(x_i,xo):
    for yi in range(y_i,yo):
        weight[xi-x_i][yi-y_i]=1-math.sqrt((xi-x0)*(xi-x0)+(yi-y0)*(yi-y0))/h  #距离中心点远近不同，权重不一
        C+=weight[xi-x_i][yi-y_i]
        quzhi=int(img[yi][xi]/16)
        sum[quzhi]+=weight[xi-x_i][yi-y_i]
C=1/C#归一化系数
for i in range(0,n):
    q[i]=C*sum[i]*255  #目标模型概率密度
roi=np.zeros((a,b))
ret,img2=cap.read()
cv2.destroyAllWindows()
while(ret):#循环直至最后一帧
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)#转换灰度
    if ret==0:
        break
    while(1):
        roi=img2[(fy-(yo-y_i)/2):(fy+(yo-y_i)/2),(fx-(xo-x_i)/2):(fx+(xo-x_i)/2)]#读取窗口内数据
        m10=0#图像矩初始化
        m01=0
        m00=0
        for i in range(0,roi.shape[0]):#计算图像矩
            for j in range(0,roi.shape[1]):
                #ax=int(roi[i][j]/40)*40
                #ax=q[int(ax/16)]
                ax=roi[i][j]
                m10+=j*ax
                m01+=i*ax
                m00+=ax
        try:
            fx_next=int(m01/m00)+fy-(yo-y_i)/2#用图像矩计算重心
            fy_next=int(m10/m00)+fx-(xo-x_i)/2
            if(fx_next==fy and fy_next==fx):#判断重心、窗口中心是否重叠，重叠推出迭代
                fy=fx_next
                fx=fy_next
                break
            else:
                fy=fx_next
                fx=fy_next
        except:
            print('error')
            break
    cv2.rectangle(img2,(fx-(xo-x_i)/2,fy-(yo-y_i)/2),(fx+(xo-x_i)/2,fy+(yo-y_i)/2),(0,255,0),3)
    cv2.imshow('image',img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret,img2=cap.read()
cap.release()
cv2.destroyAllWindows()
end=time.time()
print(end-start)