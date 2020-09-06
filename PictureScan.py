import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local

def edge_detection(filename):
    """边缘检测"""
    # 缩小图片
    image = cv2.imread(filename)
    ratio = image.shape[0]/500
    orig = image.copy()
    image = imutils.resize(image, height=500)                   # 改变图片大小，保持长宽比例

    # 转为灰度图，滤波，寻找边缘
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (9,9), 0)         # 高斯滤波
    edged_image = cv2.Canny(gray_image, 75, 200)

    edged_image = cv2.rectangle(edged_image,(0,0),(edged_image.shape[1]-1,edged_image.shape[0]-1),(255,255,255),thickness=2)

    
    #cv2.imshow("edge_image", edged_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return orig, image, ratio, edged_image

def find_contour(image, edged_image):
    """寻找外轮廓"""
    # 寻找轮廓
    cnts = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:5]      # 计算轮廓面积并排序
    image_cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    area = edged_image.shape[0]*edged_image.shape[1]
    cnts = [c for c in cnts if cv2.contourArea(c)<0.98*area]

    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*perimeter, True)          # 多边形拟合

        if len(approx)==4 and cv2.contourArea(approx)>0.1*area:
            screenCnt = approx
            break
    else:
        perimeter = cv2.arcLength(image_cnt, True)
        screenCnt = cv2.approxPolyDP(image_cnt, 0.02*perimeter, True)

    #cv2.drawContours(image, [screenCnt], -1, (0,255,0), 2)
    #cv2.imshow("Outline", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return screenCnt

def order_points_new(points):
    """调整坐标顺序（顺时针）"""
    # 按照x坐标从小到大排列
    sorted_x = points[np.argsort(points[:,0]),:]

    left_most = sorted_x[:2,:]                                      # 左边的两个坐标
    right_most = sorted_x[2:,:]                                     # 右边的两个坐标
    if left_most[0,1]!=left_most[1,1]:
        left_most = left_most[np.argsort(left_most[:,1]),:]
    else:
        left_most = left_most[np.argsort(left_most[:,0])[::-1],:]
    (tl, bl) = left_most

    if right_most[0,1]!=right_most[1,1]:
        right_most = right_most[np.argsort(right_most[:,1]),:]
    else:
        right_most = right_most[np.argsort(right_most[:,0])[::-1],:]
    (tr, br) = right_most
    return np.array([tl,tr,br,bl], dtype="float32")

def four_point_transform(image, points):
    rect = order_points_new(points)
    (tl,tr,br,bl) = rect

    # 计算矩形长宽
    width_b = np.sqrt((br[0]-bl[0])**2 + (br[1]-br[0])**2)
    width_t = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
    width = max(int(width_b), int(width_t))
    height_r = np.sqrt((br[0]-tr[0])**2 + (br[1]-tr[0])**2)
    height_l = np.sqrt((bl[0]-tl[0])**2 + (bl[1]-tl[1])**2)
    height = max(int(height_r),int(height_l))

    dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype="float32")
    trans = cv2.getPerspectiveTransform(rect, dst)
    trans = cv2.warpPerspective(image, trans, (width,height))
    return trans

def OTSU(image_gray, th_begin = 0, th_end = 256, th_step = 1):
    """大津法"""
    #计算图像直方图
    bins = np.zeros(256)
    #g最大值
    g_max = 0
    #最佳阈值
    best_threshold = 0

    for threshold in range(th_begin, th_end ,th_step):
        #计算得出图中为前景像素的数组
        bin_image_one = image_gray < threshold
        #计算得出图中为背景像素的数组
        bin_image_two = image_gray >= threshold
        #计算前景、背景像素个数
        fore_pixel = np.sum(bin_image_one)
        back_pixel = np.sum(bin_image_two)

        #计算前景像素占图像比
        w0 = float(fore_pixel)/image_gray.size
        #计算前景像素平均灰度
        u0 = float(np.sum(image_gray * bin_image_one))/fore_pixel
        #计算背景像素占图像比
        w1 = float(back_pixel)/image_gray.size
        #计算背景像素平均灰度
        u1 = float(np.sum(image_gray * bin_image_two))/back_pixel

        #计算类间方差
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        #选择最佳阈值
        if g > g_max:
            g_max = g
            best_threshold = threshold
    return best_threshold

def perspective_transform(image, screenCnt, ratio):
    trans = four_point_transform(image, screenCnt.reshape(4,2)*ratio)

    trans = cv2.cvtColor(trans, cv2.COLOR_BGR2GRAY)
    threshold = OTSU(trans, th_step = 10)
    trans = cv2.threshold(trans, threshold, 255, cv2.THRESH_BINARY)

    #cv2.imshow("Scanned", imutils.resize(trans[1], height=650))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite("test_trans_5.jpg",trans[1])
    return trans[1]

def picture_scan(filename):
    origin, image, ratio, edged_image = edge_detection(filename)
    screenCnt = find_contour(image, edged_image)
    trans = perspective_transform(origin, screenCnt, ratio)
    return trans

if __name__=="__main__":
    picture_scan("./test_5.jpg")
