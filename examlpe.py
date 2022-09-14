import numpy as np
from numpy import mgrid, sum
import math
import cv2
import matplotlib.pyplot as plt


def subimage(image, center, theta, width, height):
    shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)
    #cv2.circle(image, (center), 2, (0, 0, 0), -1)

    return image

def end():
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def covariance(X, Y):
    """
    Calculate the covariance between two random variables.
    """
    xbar = np.mean(X)
    ybar = np.mean(Y)
    covar = 1 / (len(X) - 1) * np.sum( (X - xbar) * (Y - ybar) )
    return covar



img = cv2.imread("C:\img\paper\img4c.png",0)#   사용할 이미지 파일 흑백(0)
h, w = img.shape #총 가로 세로 길이 가져옴 100*100임
cv2.namedWindow("test1")
cv2.moveWindow("test1", 800, 400)
cv2.imshow('test1', img)  # 이미지 출력
cv2.waitKey(0)

x=[] #x좌표값들
y=[] #y좌표값들
#이미지의 픽셀 값이 0이 아닌 좌표값을 각각 x,y에 저장
for h in range(h):
    for w in range(w):
        val = img[h, w]
        if val != 0:
            x.append(w)
            y.append(h)


print("x리스트",x)
print("y리스트",y)


print(f"Variance for x   : {covariance(x, x)}")
print(f"Covariance of x,y: {covariance(x, y)}")

# covariance matrix
cov = np.cov(x[:],y[:], bias=True)
print("공분산 행렬")
print(cov)
print(" ")

# image

plt.scatter(x[:],y[:],c="red")
plt.axis([0, 105, 0, 105])
#plt.grid(linestyle='dotted')
plt.show()


#-----------abc활용 공분산 행렬-------------

M = cv2.moments(img)
# Mass Center of Image
cX = int(M["m10"] / M["m00"])  # Xc 오브젝트의 중심
cY = int(M["m01"] / M["m00"])  # Yc  오브젝트의 중심
#print("오브젝트 중심좌표 : ", cX, cY)

a = ((M["m20"] / M["m00"]) - cX ** 2)
b = -(2 * ((M["m11"] / M["m00"]) - (cX * cY)))
c = ((M["m02"] / M["m00"]) - cY ** 2)
theta = math.degrees(np.arctan2(b, (a - c)) / 2)
print("a, b, c : ", a, b, c)

cov_abc = [[a, b/2],
           [b/2, c]]
print(cov_abc)
print(theta)


if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()