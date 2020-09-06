# USAGE
# python main.py -v cv.mp4
# python main.py

# 导入所需的库/包
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
import time
import math

#获取滤镜颜色
def getBGR(table, b, g, r):
    #计算标准颜色表中颜色的位置坐标
    x = int(g/4 + int(b/32) * 63)
    y = int(r/4 + int((b%32) / 4) * 63)
    #返回滤镜颜色表中对应的颜色
    return table[x][y]

# 压缩后输出画面大小
compressWidth = 600
# 窗口设置
windowName = 'Beauty'
trackbar1 = 'Beauty'
trackbar2 = 'Filter'
cv.namedWindow(windowName)
cv.createTrackbar(trackbar1, windowName, 6, 10, lambda x: None)
cv.createTrackbar(trackbar2, windowName, 0, 20, lambda x: None)
# 滤镜色彩查找表读取
lj1 = cv.imread('./filter/lj1.png')
lj2 = cv.imread('./filter/lj2.png')
lj3 = cv.imread('./filter/lj3.png')
lj4 = cv.imread('./filter/lj4.png')
lj5 = cv.imread('./filter/lj5.png')
lj6 = cv.imread('./filter/lj6.png')
lj7 = cv.imread('./filter/lj7.png')
lj8 = cv.imread('./filter/lj8.png')
lj9 = cv.imread('./filter/lj9.png')
lj10 = cv.imread('./filter/lj10.png')
lj11 = cv.imread('./filter/lj11.png')
lj12 = cv.imread('./filter/lj12.png')
lj13 = cv.imread('./filter/lj13.png')
lj14 = cv.imread('./filter/lj14.png')
lj15 = cv.imread('./filter/lj15.png')
lj16 = cv.imread('./filter/lj16.png')
lj17 = cv.imread('./filter/lj17.png')

# 构造参数解析并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# 定义肤色HSV范围
faceLower = (0, 50, 145)
faceUpper = (180, 115, 230)

# 未提供视频路径则实时捕获摄像头画面
if not args.get("video", False):
	vs = VideoStream(src=0).start()
	fps = 10
	size = (compressWidth, int(compressWidth / 16 * 9) - 1)
# 提供视频路径则读取本地视频
else:
	vs = cv.VideoCapture(args["video"])
	fps = vs.get(cv.CAP_PROP_FPS)
	size = (int(vs.get(cv.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv.CAP_PROP_FRAME_HEIGHT)))
	size = (compressWidth, int(vs.get(cv.CAP_PROP_FRAME_HEIGHT)/vs.get(cv.CAP_PROP_FRAME_WIDTH)*compressWidth))

# 设置视频写入保存参数
fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')
out = cv.VideoWriter('oops.avi', fourcc, fps, size)

# 允许摄像头或视频文件加载时间
time.sleep(2.0)

# 循环获取当前帧并进行后续处理
while True:
	frame = vs.read()

	# 对于实时捕获与本地视频进行当前帧的不同处理
	frame = frame[1] if args.get("video", False) else frame

	# 判断是否已经读取到视频结尾
	if frame is None:
		print('End.')
		break

	# 缩放（压缩）当前帧，高斯模糊后转换为HSV色域
	frame = imutils.resize(frame, width=compressWidth)
	ori = frame
	blurred = cv.GaussianBlur(frame, (11, 11), 0)
	hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

	# 根据肤色范围输出二值图遮罩，进行腐蚀与膨胀操作去除噪声
	mask = cv.inRange(hsv, faceLower, faceUpper)
	mask = cv.erode(mask, None, iterations=2)
	mask = cv.dilate(mask, None, iterations=2)

	# 获取窗口拖动条值进行磨皮美颜处理
	value1 = cv.getTrackbarPos(trackbar1, windowName)
	value2 = 1
	dx = int(value1)
	fc = value1 * 2.5
	dst = frame + 2 * cv.GaussianBlur((cv.bilateralFilter(frame, dx, fc, fc) - frame + 128), (2 * value2 - 1, 2 * value2 - 1), 0, 0) - 255

	# 使用遮罩与反遮罩对磨皮美颜后结果与原始当前帧进行处理并合并为一张处理后的图片
	face = cv.add(dst, np.zeros(np.shape(frame), dtype=np.uint8), mask=mask)
	frame = cv.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=cv.bitwise_not(mask))
	frame = cv.add(face, frame)

	# 获取窗口拖动条值以及当前遍历像素点RGB色值进行滤镜处理
	for h in range(0, frame.shape[0]):
		for w in range(0, frame.shape[1]):
			b = int(frame[h, w, 0])
			g = int(frame[h, w, 1])
			r = int(frame[h, w, 2])
			i = cv.getTrackbarPos(trackbar2, windowName)
			if i == 1:
				frame[h, w, 2] = int(0.393 * r + 0.769 * g + 0.189 * b) if int(0.393 * r + 0.769 * g + 0.189 * b) < 255 else 255
				frame[h, w, 1] = int(0.349 * r + 0.686 * g + 0.168 * b) if int(0.349 * r + 0.686 * g + 0.168 * b) < 255 else 255
				frame[h, w, 0] = int(0.272 * r + 0.534 * g + 0.131 * b) if int(0.272 * r + 0.534 * g + 0.131 * b) < 255 else 255
			elif i == 2:
				frame[h, w, 2] = int(abs(g - b + g + r) * r / 256) if int(abs(g - b + g + r) * r / 256) < 255 else 255
				frame[h, w, 1] = int(abs(b - g + b + r) * r / 256) if int(abs(b - g + b + r) * r / 256) < 255 else 255
				frame[h, w, 0] = int(abs(b - g + b + r) * g / 256) if int(abs(b - g + b + r) * g / 256) < 255 else 255
			elif i == 3:
				frame[h, w, 2] = r
				frame[h, w, 1] = g
				frame[h, w, 0] = math.sqrt(b) * 12 if math.sqrt(b) * 12 < 255 else 255
			elif i == 4:
				frame[h, w] = getBGR(lj1, b, g, r)
			elif i == 5:
				frame[h, w] = getBGR(lj2, b, g, r)
			elif i == 6:
				frame[h, w] = getBGR(lj3, b, g, r)
			elif i == 7:
				frame[h, w] = getBGR(lj4, b, g, r)
			elif i == 8:
				frame[h, w] = getBGR(lj5, b, g, r)
			elif i == 9:
				frame[h, w] = getBGR(lj6, b, g, r)
			elif i == 10:
				frame[h, w] = getBGR(lj7, b, g, r)
			elif i == 11:
				frame[h, w] = getBGR(lj8, b, g, r)
			elif i == 12:
				frame[h, w] = getBGR(lj9, b, g, r)
			elif i == 13:
				frame[h, w] = getBGR(lj10, b, g, r)
			elif i == 14:
				frame[h, w] = getBGR(lj11, b, g, r)
			elif i == 15:
				frame[h, w] = getBGR(lj12, b, g, r)
			elif i == 16:
				frame[h, w] = getBGR(lj13, b, g, r)
			elif i == 17:
				frame[h, w] = getBGR(lj14, b, g, r)
			elif i == 18:
				frame[h, w] = getBGR(lj15, b, g, r)
			elif i == 19:
				frame[h, w] = getBGR(lj16, b, g, r)
			elif i == 20:
				frame[h, w] = getBGR(lj17, b, g, r)

	# 实时预览处理后结果
	cv.imshow(windowName, frame)
	key = cv.waitKey(1) & 0xFF

	# 保存当前处理后帧
	out.write(frame)

	# q为程序停止键，按下跳出循环
	if key == ord("q"):
		break

# 停止/释放视频流
if not args.get("video", False):
	vs.stop()
else:
	vs.release()

# 释放视频写入
out.release()

# 关闭所有窗口
cv.destroyAllWindows()