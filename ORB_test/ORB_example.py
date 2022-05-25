import cv2
import os


path = os.path.join('/home/frow4s/Рабочий стол/Diploma/Main/ORB_test/', 'ORB_test.png')
image = cv2.imread(path, 0)

# Создаем ORB-детектор
orb = cv2.ORB_create()

# поиск признаков с ORB
keypoints = orb.detect(image,None)

# вычисление дескрипторов ORB
keypoints, descriptors = orb.compute(image, keypoints)

# отрисовка местоположоения признака
image2 = cv2.drawKeypoints(image, keypoints, None, (0, 0, 255), 0)
cv2.imshow('ORB example', image2)
cv2.waitKey()


