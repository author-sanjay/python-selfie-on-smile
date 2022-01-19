import cv2
import datetime
cap= cv2.VideoCapture(0) #zero is for reading video from webcam
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
while True:
    _, frame = cap.read()  #reading video in frame
    original_frame = frame.copy() #creating original frame copy so that no lines come in photo
    greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #converting into grey so that face detection is easy
    face = face_cascade.detectMultiScale(greyscale, 1.3, 5) #detecting face, 1.3 and 5 denorte the intensity of face detection
    for x, y, w, h in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)  #defining dimensions of face detection square
        face_roi = frame[y:y+h, x:x+w] #giving area of intrest to search smile
        grey_roi = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(grey_roi, 1.3, 25)

        for x1, y1, w1, h1 in smile:
            cv2.rectangle(face_roi, (x1,y1), (x1+w1, y1+h1), (0,0, 255), 2) #detecting smile
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')   #for multiple selfies saves file name as time stamp
            file_name = f'selfie-{time_stamp}.png'
            cv2.imwrite(file_name, original_frame)  #saving selfie
    cv2.imshow('Sefli', frame)
    if cv2.waitKey(10) == ord('q'):
        break
