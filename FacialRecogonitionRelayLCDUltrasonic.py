import RPi.GPIO as GPIO
import RPi.GPIO as GPIO
import time
import time
import cv2
import sys
import numpy as np
from os import listdir #use to fetch data from directory
from os.path import isfile, join
import lcddriver

#set GPIO Pins
lock=17
red=4
GPIO_TRIGGER = 18
GPIO_ECHO = 24
GPIO_LED = 21

GPIO.setmode(GPIO.BCM) # Broadcom pin-numbering scheme
GPIO.setmode(GPIO.BCM)
GPIO.setup(lock, GPIO.OUT)
GPIO.setup(red, GPIO.OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_LED, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

lcd = lcddriver.lcd()


data_path = '/home/pi/Desktop/Computer Vision Raspberry /faceData/'
only_files = [f for f in listdir(data_path) if isfile(join(data_path,f))]

def distance():
    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance

Training_Data, Labels = [], [] 

for i, files in enumerate(only_files): #itterate
	image_path = data_path + only_files[i]
	images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
	Training_Data.append(np.asarray(images,dtype=np.uint8))
	Labels.append(i)

Labels = np.asarray(Labels,dtype = np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data),np.asarray(Labels))
print("Model Training Complete")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	if faces is():
		print("No face Found!")
		return img,[]

	for(x,y,w,h) in  faces:
		cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,255),2)
		#region of interest
		roi = img[y:y+h, x:x+w]
		roi = cv2.resize(roi,(200,200))
	return img,roi

cap = cv2.VideoCapture(0)
while True:
	
	ret,frame = cap.read()
	image, face = face_detector(frame)
	try:
		dist = distance()
		#print ("Measured Distance = %.1f cm" % dist)
		if dist < 58:
			GPIO.output(GPIO_LED,False)
		else:
			GPIO.output(GPIO_LED, True)
		time.sleep(1)
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		result = model.predict(face)

		if result[1]<500:
			confidence = int(100*(1-(result[1])/300))
			display_string = 'Face Matches Percentage:'+str(confidence)+"%"
			cv2.putText(image,display_string,(100, 400), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)

		if confidence >75:
			cv2.putText(image, "Welcome Back!", (200, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (0,196, 255), 2)
			cv2.imshow('Face Cropper', image)
			GPIO.output(lock, GPIO.LOW)
			GPIO.output(red, GPIO.LOW)
			lcd.lcd_clear()
			lcd.lcd_display_string("Door Unlock:", 1)

		else:
			cv2.putText(image, "Ring the Door Bell!", (200, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (155, 87, 255), 2)
			cv2.imshow('Face Cropper', image)
			GPIO.output(lock, GPIO.HIGH)
			GPIO.output(red, GPIO.LOW)
			lcd.lcd_clear()
			lcd.lcd_display_string("Ring the Door Bell!", 1)
		
	except:
		cv2.putText(image, "Face Not Found!", (200, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow('Face Cropper', image)
		GPIO.output(lock, GPIO.HIGH)
		GPIO.output(red, GPIO.HIGH)
		lcd.lcd_clear()
		lcd.lcd_display_string("Face Not Found:", 1)
	
	key = cv2.waitKey(1) & 0xFF
	if key==27:  # ESC
		GPIO.cleanup()
		print("Clean Up!")
		sys.exit()
				
cap.release()
cv2.destroyAllWindows()
