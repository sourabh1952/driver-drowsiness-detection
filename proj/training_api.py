import uuid #give an unique id to every image here a function used which is uuid1
import os   #manipulate directory from python  take the image generate by uuid and put it in path img_path
import time #define the time between capturing between taking two image
import cv2  #to giving access the camera take default color bgr 
import torch #module loder which load yoloV5
import matplotlib.pyplot as plt
from time import sleep
import tkvideo as tkv
import tkinter as tk #it provide a basic GUI  similar to HTML
from tkinter import *
import customtkinter as ctk  #similar to css
import numpy as np
import pandas as pd

from PIL import Image, ImageTk



img_path = os.path.join('datatest', 'images')
img_labels = ['awake', 'drowsy']
number_imgs = 20



api_root=tk.Tk() #framwork of app
api_root.geometry("640x608")
api_root.title("Project Accident")
ctk.set_appearance_mode("dark")
api_label=tk.Label(api_root, text="Welcome To Project Accident")
api_label.pack() #sab chizo ko pack kar ra hai

def train_data():
 camFrame=tk.Frame(height=1280,width=720)
 camFrame.pack()
 cam_vid=ctk.CTkLabel(camFrame)
 cam_vid.pack()
 capture_vid=cv2.VideoCapture(0) #excess hardware intregated device
#setting pixel size of image 640X480
 capture_vid.set(3,640)
 capture_vid.set(4,480)
# we have to genrate 20 awake and 20 drowsy image
 for label in img_labels:
  print('Image Collection for < >'.format(label))
  time.sleep(5)#switching time between awake and drowsy
	    
  for img_num in range(number_imgs):
	   
   print('Image Collection for < >, image number < >'.format(label, img_num))
					
   ret, frame = capture_vid.read()#read video from camera and render to it on screen
	#trasport image to actual memory				
   imgname = os.path.join(img_path, label+'.'+str(uuid.uuid1())+'.jpg')
						
   cv2.imwrite(imgname, frame)#saving the image
					
   cv2.imshow('Collection of Images', frame)
				
   time.sleep(1)
				
   if cv2.waitKey(10) & 0xFF == ord('q'):
    break
 capture_vid.release()#close the camera
 cv2.destroyAllWindows()#destroy tkinter video
 for label in img_labels:
  print('Image Collection for < >'.format(label))
  for img_num in range(number_imgs):
   print('Image Collection for < >, image number < >'.format(label, img_num))
   imgname = os.path.join(img_path, label+'.'+str(uuid.uuid1())+'.jpg')
   print(imgname)
   
   
train_button=tk.Button(api_root,text="Train Model for Drowsiness",command=train_data,padx=6, pady=6, bg='cyan')
train_button.pack()

api_root.mainloop()   
