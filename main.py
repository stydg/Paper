import numpy as np
import math
import cv2
np.seterr(over='ignore')

#사용자 정의 매개변수 S
S = 17
SIZE = int((S-1)/2)
D = 50
n = 9
threshold = 1/9

draw_point = []
total_strok = 0

def getItem(image, j, i):
    return [image.item(j, i, 0), image.item(j, i, 1), image.item(j, i, 2)]

def setItem(image, j, i, color):
    if depth == 1:
        image.itemset(j, i, 0, color)
    else:
        image.itemset(j, i, 0, color)
        image.itemset(j, i, 1, color)
        image.itemset(j, i, 2, color)

#START--color diff img--#
def compute_color_diff_img():
    h, w, cs = source_luv.shape #color space == cs
    stroke_area_img = np.zeros((h, w, 1), np.uint8) + 255 #최종 결과 이미지 저장
    curr_rate = 0
    work = w * h
    progress_rate = (w * h) / 10

    for y in range(0 + SIZE, h - SIZE):
        for x in range(0 + SIZE, w - SIZE):
            stroke_area_img[y, x] = colorDiffImg(y, x) #중심좌표에  덮는값은 0~255값평균임
            rate = curr_rate / work * 100
            if rate % 10 == 0:
                print("Progress :", rate)
            curr_rate += 1
    cv2.imwrite("result2.png", stroke_area_img)
    cv2.imshow("compute_color_diff_img", stroke_area_img)
    return stroke_area_img

def colorDiffImg(y, x): # x,y를 중심으로하는 local 영역의 색차이미지 계산하고 x,y의 intensity(밝기)값 반환

    # 로컬이미지 : Cw 저장공간
    local_luv = source_luv[y-SIZE: y+SIZE+1, x-SIZE: x+SIZE+1]

    #색상 : c (로컬이미지의 가운데있는 색)의 좌표
    c = SIZE

    #중심 컬러값 가져옴 (Vc, Lc, Uc) = local_luv[c, c]
    base_c = getItem(local_luv, c, c)

    sum1 = 0
    for i in range(0, S):
        for j in range(0, S):
            #(v, l, u) = local_luv[j, i]
            nc = getItem(local_luv, j, i) #각 좌표 컬러값 가져옴

            d = distance(base_c[0], base_c[1], base_c[2], nc[0], nc[1], nc[2]) #좌표에서 가져온 컬러값과 중심색의 색차를 거리로 계산
            sum1 += np.uint8(functionF(d) * 255.0)

    result = int(sum1 / (S * S))
    return result

def distance(l1, u1, v1, l2, u2, v2):
    result = math.sqrt((l2 - l1)**2 + (u2 - u1)**2 + (v2 - v1)**2) # Difference
    return result

def functionF(d):
    if 0 <= d <= D:
        distance = (1.0 - ((d / D) * (d / D)))
        colorDiff = distance * distance
        return colorDiff
    return 0
#-----------------------------------------------------------------------------END--color diff img--#

#-----------------------------------------------------------------------------START--distrib--#
def dider():
    axiom = "a"
    #plus = [int(w / ((2 ** n))), int(h / ((2 ** n)))]

    dxdy = np.array([[1, 0],  # right x가 1증가
                     [0, 1],  # down
                     [-1, 0],  # left
                     [0, -1]])  # up y가 1증가

    s = axiom  # string to iterate upon
    for i in np.arange(n):
        s = apply_rules(s)

    s = s.replace("a", "") #a 와 b 를 없에기
    s = s.replace("b", "")

    #frame_counter = 0
    p = np.array([[0, 0]])  # this is the starting point (0,0)

    # iterate on the string s
    for i, c in enumerate(s):  # i는 인덱스값 c는 + - f 중 무언가(s배열에 들어있는데로겠지 뭐)
        # print("{:d}/{:d}".format(i,len(s)))
        # "+" 시계방향 회전으로 dxdy변위 바꿈
        if c == '+': dxdy = np.roll(dxdy, +1, axis=0)
        # "-" 반시계 방향 회전으로
        if c == '-': dxdy = np.roll(dxdy, -1, axis=0)
        # forward "f"
        if c == 'f':
            p = np.vstack([p, [p[-1, 0] + dxdy[0, 0],
                               p[-1, 1] + dxdy[0, 1]]])  # 앞 배열에 뒷배열 붙여넣기 인데/(이전 p배열의 x값+ 현dxdy의 0행꺼의 x값) y값도 마찬가지

            #frame_counter += 1
    r2 = draw(p)
    r1[SIZE:h1 - SIZE, SIZE:w1 - SIZE] = r2
    cv2.imwrite("distributin2.png", r1)
    return r1

def apply_rules(s):
    s = s.replace("a", "-Bf+AfA+fB-")  # capital letters "A" and "B" so that the second operation
    s = s.replace("b", "+Af-BfB-fA+")  # doesn't apply to the changes already made
    return s.lower()  # make everyone lowercase

def draw(p):
    dotImg = np.zeros((h, w, depth), np.uint8) + 255
    sum=0
    for a in range(0, len(p)):
        i1, j1 = p[a, :] # 시작점
        if i1 < w and j1 <h:
            if r2[j1,i1,0] == 0:
                sum += 0
            else:
                sum +=(1/r2[j1,i1,0])

        if threshold <= sum:
            sum = 0
            if j1 <= h and i1 <= w:  # 힐베르트 점찍는걸 이미지 범위 넘어가기 전까지만 함
                dot(dotImg, j1, i1, 0)
                draw_point.append([j1, i1])  # y, x 순으로 저장
        total_strok = len(draw_point)

    return dotImg
       # cv2.line(r2, (i1, j1), (i2, j2), (0,255,0), thickness=1, lineType=cv2.LINE_AA)


def dot(img, i, j, val):
    setItem(img, i, j, val)

#-----------------------------------------------------------------------------END--distrib--#
def end():
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

#원본 이미지 로드
img_source = cv2.imread("C:\img\paper\pet.png", cv2.IMREAD_COLOR)

# BGR->CIE Luv로 변환
source_luv = cv2.cvtColor(img_source, cv2.COLOR_LRGB2Luv)

r1 = compute_color_diff_img()
h1, w1, depth1 = r1.shape
r2 = r1[SIZE:h1-SIZE, SIZE:w1-SIZE]
h, w, depth = r2.shape

cv2.imshow("distribution2", dider())
end()