import cv2 as cv
import base64
import os
import numpy as np
import datetime
import random
from flask import Flask, request, jsonify, Response, g, json
from PictureScan import picture_scan
from PictureJoint import picture_joint_normal, picture_joint_normal2, picture_joint_auto

# capture = cv.VideoCapture('https://store.nintendo.co.jp/client_info/CX24DMSJKX/itemimage/HAC_8_CDHWC/HAC_8_CDHWC.main01.jpg')
# img = (capture.read())[1]
# # img = cv.imread('1.JPG')
# imgb64 = base64.b64encode(img)
# print(imgb64)
# cv.imshow('img', img)
# cv.waitKey()
#-*-coding:utf-8-*-
basedir = os.path.abspath(os.path.dirname(__file__))

def create_uuid(): #生成唯一的图片的名称字符串，防止图片显示时的重名问题
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S");  # 生成当前时间
    randomNum = random.randint(0, 100);  # 生成的随机整数n，其中0<=n<=100
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum);
    uniqueNum = str(nowTime) + str(randomNum);
    return uniqueNum;


#获取滤镜颜色
def getBGR(table, b, g, r):
    #计算标准颜色表中颜色的位置坐标
    x = int(g/4 + int(b/32) * 64)
    y = int(r/4 + int((b%32) / 4) * 64)

    #返回滤镜颜色表中对应的颜色
    return table[x][y]


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'WeChat Miniprogram Pic What?!'


@app.route("/upload", methods=['POST'])
def upload():
    # 接收图片
    upload_file = request.files.get('file')
    ext = upload_file.filename.rsplit('.')[-1]
    # 获取图片名
    file_name = create_uuid() + '.' + ext
    # 文件保存目录
    file_path= './tempImages/'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        # 保存接收的图片
        upload_file.save(file_paths)
        with open(file_paths, 'rb') as f:
            b64 = base64.b64encode(f.read())
        b64str = bytes.decode(b64)
        res = {'b64': b64str, 'path': file_paths}
        res = json.dumps(res)
        print(res)
        return res


@app.route('/beauty', methods=['POST'])
def beauty():
    print(request.values)
    # 从微信小程序前端接收数据
    url = str(json.loads(request.values.get("imgTempUrl")))
    options = int(json.loads(request.values.get("beautyStrength")))
    print(url, options)

    # 使用VideoCapture获取网络图片
    # capture = cv.VideoCapture(url)
    # frame = (capture.read())[1]
    frame = cv.imread(url)
    print(frame.shape)

    # 定义肤色HSV范围
    faceLower = (0, 10, 145)
    faceUpper = (180, 115, 255)

    # 缩放（压缩）当前帧，高斯模糊后转换为HSV色域
    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    # 根据肤色范围输出二值图遮罩，进行腐蚀与膨胀操作去除噪声
    mask = cv.inRange(hsv, faceLower, faceUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    # 获取窗口拖动条值进行磨皮美颜处理
    value1 = options
    value2 = 1
    dx = int(value1)
    fc = value1 * 2.5
    dst = frame + 2 * cv.GaussianBlur((cv.bilateralFilter(frame, dx, fc, fc) - frame + 128),
                                      (4 * value2 - 1, 4 * value2 - 1), 0, 0) - 255

    # 使用遮罩与反遮罩对磨皮美颜后结果与原始当前帧进行处理并合并为一张处理后的图片
    face = cv.add(dst, np.zeros(np.shape(frame), dtype=np.uint8), mask=mask)
    frame = cv.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=cv.bitwise_not(mask))
    frame = cv.add(face, frame)

    file = url.rsplit('.')
    print(file)
    file_paths = '.' + file[1].replace('tempImages', 'tempDones') + '_' + str(options) + '.' + file[2]
    print(file_paths)
    cv.imwrite(file_paths, frame)

    # 转换为base64格式传回微信小程序前端
    with open(file_paths, 'rb') as f:
        b64 = base64.b64encode(f.read())

    return b64


@app.route('/identity', methods=['POST'])
def identity():
    print(request.values)
    # 从微信小程序前端接收数据
    url = str(json.loads(request.values.get("imgTempUrl")))
    rgb = dict(json.loads(request.values.get("RGB")))
    blur = int(json.loads(request.values.get("Blur")))
    print(url, rgb, blur)
    r = rgb['r']
    g = rgb['g']
    b = rgb['b']

    img = cv.imread(url)
    img = cv.resize(img, (int(600 / img.shape[0] * img.shape[1]), 600))
    blurred = cv.GaussianBlur(img, (blur, blur), 0)
    imghsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    aim = np.uint8([[img[0, 0, :]]])
    print(aim)
    if aim[0][0][0] > 180 and aim[0][0][1] > 180 and aim[0][0][2] > 180:
        return 'white'
    elif aim[0][0][2] > 150 and aim[0][0][0] < 80 and aim[0][0][1] < 80:
        return 'red'
    aimhsv = cv.cvtColor(aim, cv.COLOR_BGR2HSV)
    mask = cv.inRange(imghsv, np.array([aimhsv[0, 0, 0] - 10, 40, 40]), np.array([aimhsv[0, 0, 0] + 10, 255, 255]))
    mask = cv.erode(mask, None, iterations=int(19/blur))
    mask = cv.dilate(mask, None, iterations=int(19/blur))
    dismask = cv.bitwise_not(mask)
    img1 = cv.bitwise_and(img, img, mask=dismask)
    bg = img.copy()
    rows, cols, channels = img.shape
    bg[:rows, :cols, :] = [b, g, r]
    img2 = cv.bitwise_and(bg, bg, mask=mask)
    result = cv.add(img1, img2)

    file = url.rsplit('.')
    print(file)
    file_paths = '.' + file[1].replace('tempImages', 'tempDones') + '_' + str(r) + str(g) + str(b) + '.' + file[2]
    print(file_paths)
    cv.imwrite(file_paths, result)

    # 转换为base64格式传回微信小程序前端
    with open(file_paths, 'rb') as f:
        b64 = base64.b64encode(f.read())
    b64str = bytes.decode(b64)

    return b64

@app.route('/film', methods=['POST'])
def film():
    # 从微信小程序前端接收数据
    url = str(json.loads(request.values.get("imgTempUrl")))
    index = int(json.loads(request.values.get("index")))
    print(url, index)

    # 滤镜色彩查找表读取
    lj1 = cv.imread('./Filter/250D.png')
    lj2 = cv.imread('./Filter/500D.png')
    lj3 = cv.imread('./Filter/5205.png')
    lj4 = cv.imread('./Filter/5218.png')
    lj5 = cv.imread('./Filter/F125.png')
    # lj6 = cv.imread('./Filter/FilmS.png')

    frame = cv.imread(url)
    frame = cv.resize(frame, (int(600 / frame.shape[0] * frame.shape[1]), 600))
    print(frame.shape)

    # 获取窗口拖动条值以及当前遍历像素点RGB色值进行滤镜处理
    for h in range(0, frame.shape[0]):
        for w in range(0, frame.shape[1]):
            b = int(frame[h][w][0])
            g = int(frame[h][w][1])
            r = int(frame[h][w][2])
            i = index
            if i == 1:
                frame[h][w] = getBGR(lj1, b, g, r)
            elif i == 2:
                frame[h][w] = getBGR(lj2, b, g, r)
            elif i == 3:
                frame[h][w] = getBGR(lj3, b, g, r)
            elif i == 4:
                frame[h][w] = getBGR(lj1, b, g, r)
            elif i == 5:
                frame[h][w] = getBGR(lj2, b, g, r)
            # elif i == 6:
            #     frame[h][w] = getBGR(lj3, b, g, r)

    file = url.rsplit('.')
    print(file)
    file_paths = '.' + file[1].replace('tempImages', 'tempDones') + '_' + str(r) + str(g) + str(b) + '.' + file[2]
    print(file_paths)
    cv.imwrite(file_paths, frame)

    # 转换为base64格式传回微信小程序前端
    with open(file_paths, 'rb') as f:
        b64 = base64.b64encode(f.read())
    b64str = bytes.decode(b64)

    return b64


def generate_random_str(randomlength=16):
  """
  生成一个指定长度的随机字符串
  """
  random_str = ''
  base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
  length = len(base_str) - 1
  for _ in range(randomlength):
    random_str += base_str[random.randint(0, length)]
  return random_str


@app.route('/scan', methods=['POST'])
def scanupload():
    print('upload...')
    img = request.files.get('file')
    #print(type(img))
    path = basedir+"/static/photo/"
    if os.path.exists(path)==False:
        os.makedirs(path)
    file_path = path + img.filename
    img.save(file_path)
    print("图片上传成功")

    trans = picture_scan(file_path)                 # 扫描图片
    print(file_path)
    cv.imwrite(file_path,trans)                    # 写入

    with open(file_path,'rb') as f:
        img = base64.b64encode(f.read())
    #img = base64.b64encode(img)
    print(file_path[file_path.rfind('.')+1:].lower())

    # 根据图片类型转换
    if file_path[file_path.rfind('.')+1:].lower()=='jpg':
        response = Response(img, content_type='image/png')
    # elif file_path[file_path.rfind('.')+1:].lower()=='jpeg':
    #     response = Response(img, content_type='image/png')
    elif file_path[file_path.rfind('.')+1:].lower()=='png':
        response = Response(img, content_type='image/jpg')

    return response


@app.route('/upload/multi',methods=['POST'])
def upload_multi():
    print('upload/multi...')
    img = request.files.get('photo')
    path = basedir+"/static/photo/"
    if os.path.exists(path)==False:
        os.makedirs(path)
    file_path = path + img.filename
    img.save(file_path)
    #g._messages.append(file_path)
    #print(g._messages)
    print("图片上传成功")
    # 返回图片存储路径
    return jsonify({'filepath':file_path})


@app.route('/joint/auto',methods=['GET'])
def joint_auto():
    photo_list = request.args.get('data')
    #print(type(photo_list))
    print(photo_list[2:-2].split('","'))
    photo_list = photo_list[2:-2].split('","')

    img = picture_joint_auto(photo_list)
    file_path = basedir+"/static/photo/"+generate_random_str()+".jpg"
    cv.imwrite(file_path,img)

    with open(file_path,'rb') as f:
        img = base64.b64encode(f.read())
    #img = base64.b64encode(img)
    #print(file_path[file_path.rfind('.')+1:].lower())
    if file_path[file_path.rfind('.')+1:].lower()=='jpg':
        response = Response(img, content_type='image/png')
    elif file_path[file_path.rfind('.')+1:].lower()=='png':
        response = Response(img, content_type='image/jpg')
    return response

@app.route('/joint/normal',methods=['GET'])
def joint_normal():
    photo_list = request.args.get('data')
    #print(type(photo_list))
    print(photo_list[2:-2].split('","'))
    photo_list = photo_list[2:-2].split('","')

    img = picture_joint_normal(photo_list)
    file_path = basedir+"/static/photo/"+generate_random_str()+".jpg"
    cv.imwrite(file_path,img)

    with open(file_path,'rb') as f:
        img = base64.b64encode(f.read())
    #img = base64.b64encode(img)
    #print(file_path[file_path.rfind('.')+1:].lower())
    if file_path[file_path.rfind('.')+1:].lower()=='jpg':
        response = Response(img, content_type='image/png')
    elif file_path[file_path.rfind('.')+1:].lower()=='png':
        response = Response(img, content_type='image/jpg')
    return response


@app.route('/joint/normal2',methods=['GET'])
def joint_normalH():
    photo_list = request.args.get('data')
    #print(type(photo_list))
    print(photo_list[2:-2].split('","'))
    photo_list = photo_list[2:-2].split('","')

    img = picture_joint_normal2(photo_list)
    file_path = basedir+"/static/photo/"+generate_random_str()+".jpg"
    cv.imwrite(file_path,img)

    with open(file_path,'rb') as f:
        img = base64.b64encode(f.read())
    #img = base64.b64encode(img)
    #print(file_path[file_path.rfind('.')+1:].lower())
    if file_path[file_path.rfind('.')+1:].lower()=='jpg':
        response = Response(img, content_type='image/png')
    elif file_path[file_path.rfind('.')+1:].lower()=='png':
        response = Response(img, content_type='image/jpg')
    return response


if __name__ == '__main__':
    app.debug = True
    app.run()