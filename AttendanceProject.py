import cv2                          #Importing necessary libraries
import numpy as np
import face_recognition
import os
from datetime import datetime


path ='ImageAttendance'        #Folder which contains images of people to detect
                               #To detect more faces , add the images of those people in this folder.
images=[]
classNames=[]

myList=os.listdir(path)
print(myList)

for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):              #Function to find Encodings
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # face_recognition library works on RGB images
                                                           # dissimilar to BGR images used by a Computer

        encode=face_recognition.face_encodings(img)[0]      #finding image encodings
        encodeList.append(encode)

    return encodeList

def markAttendance(name):                   #Function to mark Attendance and store it in a csv or Excel File.
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])

        if name not in nameList:             #To prevent marking Attendance twice
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')    #storing the name and time in the Attendance File.



encodeListKnown=findEncodings(images)           #Function to find encodings of images called.

# print("Encoding Complete")

cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)           #resizing the input of webcam to fasten the processing

    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace , faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)      #comparing the faces in webcam from images (bool value)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)      #Comparing the encodings to recognize faces.
        # print(faceDis)
        matchIndex=np.argmin(faceDis)                                           #Minimum face distance image is the most similar image found .

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4                                 #Changing coordinates with respect to original webcam image

            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,122),3)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,54,122),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(244,0,23),2)           #Writing the name of the recognized face

            markAttendance(name)   #Function to mark Attendance called.

    cv2.imshow("Webcam",img)
    cv2.waitKey(1)




### Made by : PRAKARTI GOEL ###
