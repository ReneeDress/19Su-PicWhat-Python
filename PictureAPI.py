from flask import Flask, request, jsonify, Response, g
import base64
import os
from cv2 import cv2
import random
from PictureScan import picture_scan
from PictureJoint import picture_joint_normal, picture_joint_auto

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

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
def upload():
    print('upload...')
    img = request.files.get('file')
    #print(type(img))
    path = basedir+"\\static\\photo\\"
    if os.path.exists(path)==False:
        os.makedirs(path)
    file_path = path + img.filename
    img.save(file_path)
    print("图片上传成功")

    trans = picture_scan(file_path)                 # 扫描图片
    cv2.imwrite(file_path,trans)                    # 写入

    with open(file_path,'rb') as f:
        img = base64.b64encode(f.read())
    #img = base64.b64encode(img)
    #print(file_path[file_path.rfind('.')+1:].lower())
    # 根据图片类型转换
    if file_path[file_path.rfind('.')+1:].lower()=='jpg':
        response = Response(img, content_type='image/png')
    elif file_path[file_path.rfind('.')+1:].lower()=='png':
        response = Response(img, content_type='image/jpg')
    return response

@app.route('/upload/multi',methods=['POST'])
def upload_multi():
    print('upload/multi...')
    img = request.files.get('photo')
    path = basedir+"\\static\\photo\\"
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
    #print(photo_list[2:-2].split('","'))
    photo_list = photo_list[2:-2].split('","')

    img = picture_joint_auto(photo_list)
    file_path = basedir+"\\static\\photo\\"+generate_random_str()+".jpg"
    cv2.imwrite(file_path,img)

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
    file_path = basedir+"\\static\\photo\\"+generate_random_str()+".jpg"
    cv2.imwrite(file_path,img)

    with open(file_path,'rb') as f:
        img = base64.b64encode(f.read())
    #img = base64.b64encode(img)
    #print(file_path[file_path.rfind('.')+1:].lower())
    if file_path[file_path.rfind('.')+1:].lower()=='jpg':
        response = Response(img, content_type='image/png')
    elif file_path[file_path.rfind('.')+1:].lower()=='png':
        response = Response(img, content_type='image/jpg')
    return response

if __name__=="__main__":
    app.run(debug=True)