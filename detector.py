import cv2
import sqlite3


faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingData.yml")
font = cv2.FONT_HERSHEY_SIMPLEX


def get_profile(id):
    conn = sqlite3.connect("FaceBase")
    cmd = "Select * FROM People WHERE ID = " + str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 2, 5)
    for(x, y, w, h) in faces:
        Id = 0
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        Id, conf = rec.predict(gray[y:y + h, x:x + w])
        profile = get_profile(Id)
        if(profile is not None):
            if(Id == int(profile[0])):
                cv2.putText(img, str(profile[1]), (
                    x + 5, y + h + 20), font, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, str(profile[2]), (
                    x + 5, y + h + 40), font, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, str(profile[3]), (
                    x + 5, y + h + 60), font, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, str(profile[4]), (
                    x + 5, y + h + 80), font, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "Unknown", (
                    x + 5, y + h + 20), font, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1) == ord('q')):
        break


cam.release()
cv2.destroyAllWindows()
