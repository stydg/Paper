import numpy as np
import math
import cv2
np.seterr(over='ignore')

#사용자 정의 매개변수 S
S = 17
SIZE = int((S-1)/2)
D = 50

n = 9
threshold = 1/5
draw_point = []
count = 0
total_strok = 0
Attributes = []

def getItem(image, j, i):
    return [image.item(j, i, 0), image.item(j, i, 1), image.item(j, i, 2)]

def setItem(image, j, i, color):
    image.itemset(j, i, 0, color)
    image.itemset(j, i, 1, color)
    image.itemset(j, i, 2, color)



def end():
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def paint():
    for i in range(h1):
        for j in range(w1):
            if (r1[i,j] == [0,0,0]).all:
                draw_point.append([i,j])

    total_strok = len(draw_point)
    print(total_strok)
    for index in range(total_strok):
        print(draw_point)
        y = draw_point[index][0]
        x = draw_point[index][1]
        print(y, x)
        local_colorDiffImg(y, x)





def local_colorDiffImg(y, x): # x,y를 중심으로하는 local 영역의 색차이미지 계산하고 x,y의 intensity(밝기)값 반환

    # 로컬이미지 : Cw 저장공간
    local_luv = source_luv[y-SIZE: y+SIZE+1, x-SIZE: x+SIZE+1]
    colordiff_result_img = np.zeros(local_luv, np.uint8) + 255  # 최종 결과 이미지 저장

    #색상 : c (로컬이미지의 가운데있는 색)의 좌표
    c = SIZE

    #중심 컬러값 가져옴 (Vc, Lc, Uc) = local_luv[c, c]
    print(source_luv[y-SIZE: y+SIZE+1, x-SIZE: x+SIZE+1])
    base_c = getItem(local_luv, c, c)

    for i in range(0, S):
        for j in range(0, S):
            #(v, l, u) = local_luv[j, i]
            nc = getItem(local_luv, j, i) #각 좌표 컬러값 가져옴

            d = distance(base_c[0], base_c[1], base_c[2], nc[0], nc[1], nc[2]) #좌표에서 가져온 컬러값과 중심색의 색차를 거리로 계산
            setItem(colordiff_result_img, i, j, int(functionF(d) * 255.0))

    M = cv2.moments(colordiff_result_img)

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

    if w < l:
        w = int((M["m00"] / 255) / l)
    cv2.imshow("ted", colordiff_result_img)



def distance(l1, u1, v1, l2, u2, v2):
    result = math.sqrt((l2 - l1)**2 + (u2 - u1)**2 + (v2 - v1)**2) # Difference
    return result

def functionF(d):
    if 0 <= d <= D:
        distance = (1.0 - ((d / D) * (d / D)))
        colorDiff = distance * distance
        return colorDiff
    return 0

#원본 이미지 로드
img_source = cv2.imread("C:\img\paper\img8a.png", cv2.IMREAD_COLOR)
# BGR->CIE Luv로 변환
source_luv = cv2.cvtColor(img_source, cv2.COLOR_LRGB2Luv)

r1 = cv2.imread("distributin.png", 1)
h1, w1, depth1 = r1.shape
paint()



end()