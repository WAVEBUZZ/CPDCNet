import cv2

# 读取图像
image = cv2.imread("./VOCdevkit/VOC2007/JPEGImages/0002.jpg")

# # 定义边界框坐标
xmin, ymin, xmax, ymax = 0,120,160,190
xmin1, ymin1, xmax1, ymax1 = 125,0,256,256
# xmin2, ymin2, xmax2, ymax2 = 225,13,256,50
# xmin3, ymin3, xmax3, ymax3 = 160,0,190,90
# xmin4, ymin4, xmax4, ymax4 = 127,120,256,148

# 绘制边界框
color = (0, 255, 0)  # 绿色
thickness = 2  # 边界框线条宽度
cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
cv2.rectangle(image, (xmin1, ymin1), (xmax1, ymax1), color, thickness)
# cv2.rectangle(image, (xmin2, ymin2), (xmax2, ymax2), color, thickness)
# cv2.rectangle(image, (xmin3, ymin3), (xmax3, ymax3), color, thickness)
# cv2.rectangle(image, (xmin4, ymin4), (xmax4, ymax4), color, thickness)
# 保存包含边界框的图像
cv2.imwrite("./VOCdevkit/VOC2007/bianjie/0002.jpg",image)
