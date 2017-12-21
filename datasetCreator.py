import cv2
import sqlite3

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)


def insert_record(Id, Name, Age, Gender, Comment):
    conn = sqlite3.connect("FaceBase")
    cmd = "SELECT * FROM People WHERE ID=" + str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if (isRecordExist == 1):
        cmd = "UPDATE People SET Name=" + str(Name) + "WHERE ID=" + str(Id)
    else:
        cmd = "INSERT INTO People(ID,Name,Age,Gender, Comment) Values(" + str(
            Id) + "," + str(Name) + "," + str(Age) + "," + str(
            Gender) + "," + str(Comment) + ")"
    conn.execute(cmd)
    conn.commit()
    conn.close()


Id = input('Enter user id: ')
name = input('Enter the user name: ')
age = input('Enter the user age: ')
gender = input('Enter the user gender: ')
comment = input('Enter comment about the user: ')
insert_record(Id, name, age, gender, comment)
sampleNumber = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 2, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        sampleNumber = sampleNumber + 1
        cv2.imwrite("dataSet/User." + Id + "." +
                    str(sampleNumber) + ".jpg", gray[y:y + h, x:x + w])
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if(sampleNumber > 20):
        break

cam.release()
cv2.destroyAllWindows()
