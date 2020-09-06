import imutils
from cv2 import cv2
import numpy as np
import time


def picture_joint_normal(photo_list):
    """图片拼接"""
    photo_joint = cv2.imread(photo_list[0])
    for i in range(1, len(photo_list)):
        photo = cv2.imread(photo_list[i])
        # 调整宽度
        if photo_joint.shape[1]>photo.shape[1]:
            photo = imutils.resize(photo, width= photo_joint.shape[1])
        else:
            photo_joint = imutils.resize(photo_joint, width=photo.shape[1])
        # 图片拼接
        photo_joint = cv2.vconcat([photo_joint, photo])
    return photo_joint


def picture_joint_normal2(photo_list):
    """图片拼接_横向"""
    photo_joint = cv2.imread(photo_list[0])
    for i in range(1, len(photo_list)):
        photo = cv2.imread(photo_list[i])
        # 调整宽度
        if photo_joint.shape[0]>photo.shape[0]:
            photo = imutils.resize(photo, height= photo_joint.shape[0])
        else:
            photo_joint = imutils.resize(photo_joint, height=photo.shape[0])
        # 图片拼接
        photo_joint = cv2.hconcat([photo_joint, photo])
    return photo_joint


def compare(img1, img2):
    """直方图比较两张图片是否相同"""
    # 获取图片灰度值分布直方图
    hist1 = cv2.calcHist([img1], [0], None, [256], [0,255])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0,255])
    if (hist1==hist2).all():
        return True
    else:
        return False


def picture_joint_auto(photo_list, height=15, ratio=0.2):
    """
    图片自动拼接
    height: 重复区域的高度
    ratio: 图片缩小比例
    """
    photo_joint = cv2.imread(photo_list[0])
    for i in range(1,len(photo_list)):
        photo = cv2.imread(photo_list[i])
        # 调整宽度
        if photo_joint.shape[1]>photo.shape[1]:
            photo = imutils.resize(photo, width=photo_joint.shape[1])
        else:
            photo_joint = imutils.resize(photo_joint, width=photo.shape[1])

        # 灰度图，以一定比例缩小图片
        photo_joint_gray = cv2.cvtColor(photo_joint, cv2.COLOR_BGR2GRAY)
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        photo_joint_gray_min = cv2.resize(photo_joint_gray,(int(photo_joint_gray.shape[1]*ratio),int(photo_joint_gray.shape[0]*ratio)), interpolation=cv2.INTER_NEAREST)
        photo_gray_min = cv2.resize(photo_gray,(int(photo_gray.shape[1]*ratio),int(photo_gray.shape[0]*ratio)), interpolation=cv2.INTER_NEAREST)
        #cv2.imwrite("img2.jpg",photo_gray_min)
        #cv2.waitKey(0)

        flag = False

        for loc1 in range(photo_joint_gray_min.shape[0]-1, -1, -1):         # 图1从下往上
            temp1 = photo_joint_gray_min[loc1-height+1:loc1+1, 0:photo_joint_gray_min.shape[1]]
            for loc2 in range(0,int(photo_gray_min.shape[0]/2)):            # 图2从上往下
                temp2 = photo_gray_min[loc2:loc2+height, 0:photo_gray_min.shape[1]]
                if compare(temp1, temp2):
                    print(loc1,loc2)
                    cut_height = height
                    loc1-=height
                    loc2-=1

                    # 在原灰度图中找到最大重复区域
                    loc1 = int(loc1/ratio)
                    loc2 = int(loc2/ratio)
                    while loc1!=-1:
                        if (photo_joint_gray[loc1]==photo_gray[loc2]).all():
                            loc1-=1
                            loc2-=1
                            cut_height+=1
                        else:
                            break
                    # 图片裁剪，拼接
                    photo_joint = photo_joint[0:loc1,0:photo_joint.shape[1]]
                    photo = photo[loc2+1:photo.shape[0],0:photo.shape[1]]
                    photo_joint = cv2.vconcat([photo_joint,photo])
                    flag = True
                    break
            if flag:
                break
        # 若无重复区域，直接拼接
        else:
            photo_joint = cv2.vconcat([photo_joint,photo])
    return photo_joint


if __name__=="__main__":
    start = time.time()
    photo_list = ['./photo/t1.JPG','./photo/t2.JPG','./photo/t3.JPG']
    #picture_joint_normal(photo_list)
    picture_joint_auto(photo_list, 15, 0.2)
    print(time.time()-start)