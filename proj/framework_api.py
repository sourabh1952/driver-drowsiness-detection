import streamlit as st
import cv2
import torch
from PIL import Image 
import numpy as np 
import time
from ultralytics import YOLO
# from playsound import playsound
# for html 
st.markdown("""
<div >
Hi, this is our project
</div>

""", unsafe_allow_html=True)

html_string = """
            <audio controls autoplay>
              <source src="https://www.orangefreesounds.com/wp-content/uploads/2022/04/Small-bell-ringing-short-sound-effect.mp3" type="audio/mp3">
            </audio>
            """

FRAME_WINDOW = st.image([]) #frame window


st.markdown("""<h2 style='text-align: center;'>Here I want to position the camera.</h2>""", unsafe_allow_html=True) #title

run = st.checkbox("Start") #checkbox
#capture video 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/HP/Desktop/driver-drowsy-detection/Automatic_Alarm_for_Driver_Drowsiness_Detection/proj/best.pt', force_reload=True)
 
if run == True: # frame will render 
    capture_vid=cv2.VideoCapture(0) # capturing the video 
    capture_vid.set(3,640)
    capture_vid.set(4,480)
    while True:
        ret,frame = capture_vid.read()
        frame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #convert bgr to rgb format 
        results=model(frame) # fitting the model 
        print(results)
        x=str(results) 
        a=x[21:27]
        if(a=="drowsy"):
           sound = st.empty()
           sound.markdown(html_string, unsafe_allow_html=True) 
           time.sleep(2)  # wait for 2 seconds to finish the playing of the audio
           sound.empty()  # optionally delete the element afterwards
        img= np.squeeze(results.render()) # rgb breakdown of image 
        img= Image.fromarray(img) # rendering the image by combining the three components of color 
        print(img)
        FRAME_WINDOW.image(img) # adding the image in frame window after complete processing 



