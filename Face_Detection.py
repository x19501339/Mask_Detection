from json.tool import main
from turtle import delay
from unicodedata import name
from time import sleep
import cv2
import os
from playsound import playsound
import msvcrt, winsound
import win10toast
import numpy as np
import multiprocessing
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model

#Load some trained data on face frontals from opencv (hear cascade algorthim)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
trained_face_data_with_mask = cv2.CascadeClassifier("./Train")

#capture video with webcam
webcam = cv2.VideoCapture(0)

#load keras model
model = load_model("mask_recog_ver2.h5")

#setting up the notifation for when user is detected without a mask
toaster = win10toast.ToastNotifier()

#setting up the sound for when user is detected witout a mask
executed = False
def alert(run):
    global executed
    if (executed == False):
        print("No Mask")
        executed = True
        winsound.PlaySound('PleasePutOnAMask.wav', winsound.SND_FILENAME| winsound.SND_ASYNC,) 
        #toaster.show_toast("User Detected without a mask")

#loops over every frame in the video forever
while True:

    #read the current frame
    successful_frame_read, frame = webcam.read()

    #must conver to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    faces_list=[]
    preds=[]
    for (x, y, w, h) in face_coordinates:
        face_frame = frame[y:y+h,x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame =  preprocess_input(face_frame)

        preds = model.predict([face_frame])
        (mask, withoutMask) = preds[0]

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y- 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
 
        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)

        #if no mask is detected play "please put on a mask"
        if label[0:7] == "No Mask":
            alert(False)
            break
        elif label[0:4] == "Mask":
            executed = False
            print(label)
            
    #draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 5)

    #dislay the images with the faces
    cv2.imshow("Face Detection",frame)
    key = cv2.waitKey(1)

    #stop if Q key is pressed, every key on the keybored has a number, upper and lowercase (ASCII code)
    if key==81 or key==113:
        break
    
#release the video videocapture object
webcam.release()
