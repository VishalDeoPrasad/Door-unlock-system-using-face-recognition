import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def face_extractor(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	if faces is():
		print('Face Not Found!')
		return None

	for(x,y,w,h) in  faces:
		cropped_face = img[y:y+h, x:x+w]

	return cropped_face


count = 0
while True:
	ret, frame = cap.read()

	if face_extractor(frame) is not None:
		count+=1
		face = cv2.resize(face_extractor(frame),(200,200))
		face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

		file_name_path = 'F:/Artificial Intelligence/Computer Vision/Computer Vision Project/face data/face'+str(count)+'.jpg'
		cv2.imwrite(file_name_path,face)

		cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
		cv2.imshow('face cropped',face)
	else:
		print("Face Not Found!")
		pass

	if cv2.waitKey(1) == 13 or count == 150:
		break
cap.release()
cv2.destroyAllWindows()
print('Face Sample Collected!!')