import cv2
import os


path1 = os.path.join('/home/frow4s/Рабочий стол/Diploma/Main/FLANN_test/', '000000.png')
path2 = os.path.join('/home/frow4s/Рабочий стол/Diploma/Main/FLANN_test/', '000001.png')
image1 = cv2.imread(path1, 0)
image2 = cv2.imread(path2, 0)

# Создаем ORB-детектор
detector = cv2.ORB_create()

# Создаем FLANN-мэтчер
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

# Находим ключевые точки и их дескрипторы

kp1, des1 = detector.detectAndCompute(image1, None)
kp2, des2 = detector.detectAndCompute(image2, None)

# Находим совпадения
matches = matcher.knnMatch(des1, des2, k=2)

# Находим совпадения, которые находятся на удовлетворительном расстоянии

good = []
try:
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
except ValueError:
    pass

# Отрисовка сопоставления признаков

draw_params = dict(matchColor = -1, # draw matches in green color
                    singlePointColor = None,
                    matchesMask = None, # draw only inliers
                    flags = 2)

img3 = cv2.drawMatches(image1, kp1, image2, kp2, good , None, **draw_params)
cv2.imshow("image", img3)
cv2.waitKey()

