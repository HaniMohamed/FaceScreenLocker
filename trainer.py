import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'


def get_images_with_id(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        Ids.append(Id)
        cv2.imshow("traininng", faceNp)
        cv2.waitKey(10)
    return np.array(Ids), faces


Ids, faces = get_images_with_id(path)
recognizer.train(faces, Ids)
recognizer.write('recognizer/trainingData.yml')
cv2.destroyAllWindows()
