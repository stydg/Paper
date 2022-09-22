import numpy as np
from numpy import mgrid, sum
import math
import cv2


#사용자 정의 매개변수 S
S = 25
SIZE = int((S-1)/2)


def getItem(image, j, i):
    return [image.item(j, i, 0), image.item(j, i, 1), image.item(j, i, 2)]

def setItem(image, j, i, color):
    image.itemset(j, i, 0, color)
    image.itemset(j, i, 1, 0)
    image.itemset(j, i, 2, 0)

def subimage(image, center, theta, width, height):
    shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)
    cv2.circle(image, (center), 2, (0, 0, 0), -1)

    return image

def end():
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


img = cv2.imread("C:\img\paper\img6d.png",0)#   사용할 이미지 파일 흑백(0)
rows, cols = img.shape

flag = 0        # 스케일을 줄이고 키우고 줄이기 위한 플레그
scale = 1       # 초기값설정
angle = 0       # 초기 0도
while True:

    MT = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale) #어파인 핼렬 뱉음
    dst = cv2.warpAffine(img, MT, (cols, rows)) #img를 어파인행렬대로 이미지 회전 변환 반환
    cv2.namedWindow("test1")
    cv2.moveWindow("test1", 800, 400)
    cv2.imshow('test1', dst)  # 이미지 출력
    M = cv2.moments(dst)

    # Mass Center of Image
    cX = int(M["m10"] / M["m00"])  # Xc 오브젝트의 중심
    cY = int(M["m01"] / M["m00"])  # Yc  오브젝트의 중심
    print("오브젝트 중심좌표 : ", cX, cY)

    a = ((M["m20"] / M["m00"]) - cX ** 2)
    b = 2 * ((M["m11"] / M["m00"]) - cX * cY)
    c = ((M["m02"] / M["m00"]) - cY ** 2)
    print("a, b, c : ", a, b, c)

    # theta = math.degrees((math.atan2(b,a-c)/2)) 이건 -pi/2 ~ pi/2
    theta = math.degrees(np.arctan2(b, (a - c)) / 2)
    w = (math.sqrt(6 * (a + c - math.sqrt(b ** 2 + (a - c) ** 2))))
    l = (math.sqrt(6 * (a + c + math.sqrt(b ** 2 + (a - c) ** 2))))
    print("쎄타 w l : ", theta, w, l)

    change = True
    if w < l:
        w = int((M["m00"] / 255) / l)

    tta = math.radians(theta)
    omg = (a - c) * math.cos(2 * tta) + b * math.sin(2 * tta)
    print("이계도함수 결과", omg)


    result = np.zeros(dst.shape)
    rot_rectangle = ((cX, cY), (l, w), theta)
    box = cv2.boxPoints(rot_rectangle)
    box = np.int0(box)  # Convert into integer values
    rectangle = cv2.drawContours(result, [box], 0, (255, 255, 255), -1)
    cv2.circle(result, (cX, cY), 2, (0), -1)
    cv2.namedWindow("test2")
    cv2.moveWindow("test2", 900, 400)
    cv2.imshow("test2", rectangle) #각도 반영 안된 사각형


    # image = subimage(result, center=(cX, cY), theta=theta, width=w, height=l)
    #
    #
    # cv2.namedWindow("test3")
    # cv2.moveWindow("test3", 1000, 400)
    # cv2.imshow("test3", image)
    cv2.waitKey(0)

    angle = angle + 10     # 각을 10도 씩 회전
    if angle >= 360:        # 0도~360도
        angle = 0

    if cv2.waitKey(50) == 27:   #   esc 누르면 종료
        break

cv2.destroyAllWindows()