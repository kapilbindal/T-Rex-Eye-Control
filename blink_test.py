import cv2
import dlib
import numpy as np
import pyautogui as pg
from math import hypot


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

right_eye_arr = [36,37,38,39,40,41]
left_eye_arr = [42,43,44,45,46,47]

def get_mid(point1,point2):
	return int((point1.x+point2.x)/2),int((point1.y+point2.y)/2)
	
def check_blinking(frame,eye_points,face_landmark):

	leftPt = (face_landmark.part(eye_points[0]).x,face_landmark.part(eye_points[0]).y)
	rightPt = (face_landmark.part(eye_points[3]).x,face_landmark.part(eye_points[3]).y)
	centerTop = get_mid(face_landmark.part(eye_points[1]),face_landmark.part(eye_points[2]))
	centerBottom = get_mid(face_landmark.part(eye_points[5]),face_landmark.part(eye_points[4]))

	vertLen = hypot((centerTop[0]-centerBottom[0]),(centerTop[1]-centerBottom[1]))
	horLen = hypot((leftPt[0]-rightPt[0]),(leftPt[1]-rightPt[1]))

	ratio = horLen/vertLen 	
	return ratio

isRunning = False
while True:

	_,frame = cap.read()
	frame = cv2.flip(frame, 1)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = detector(gray)
	if cv2.waitKey(1) == ord('s'):
		isRunning = not isRunning

	for face in faces:

		landmarks = predictor(gray,face) 

		x = landmarks.part(30).x
		y = landmarks.part(30).y
		cv2.circle(frame,(x,y),6,(255,0,0),-1) 

		right_ratio = check_blinking(frame,right_eye_arr,landmarks)
		left_ratio = check_blinking(frame,left_eye_arr,landmarks)	
		
		if(left_ratio+right_ratio)/2 >5.0:
			if isRunning:
				pg.press('up')
		
		cv2.putText(frame, str(isRunning),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
		
	cv2.imshow("Frame",frame)


	key = cv2.waitKey(1)

	if key == 27:
		break

