import numpy as np
import cv2,os


print "Start Init components..."
print "opencv version:",cv2.__version__

root = "data"

#init training data set and label set
training_data  = []
training_label = []
user_label = []
counter = 0

#init the face detector, can only detect if image contains face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#init the recognizer, can recognize and identify person
recognizer = cv2.createLBPHFaceRecognizer()

#iterate image in data folder. the subfolder name is the person's name
for subdir,dirs,file in os.walk(root):

	# for each image we add to train_data
	for f in file:
		
		#string label of that person
		name = subdir.split('\\')
		name = name[1]

		#data of that image
		file_sample =  os.path.join(subdir,f)

		print "Proccesing image for",name
		
		#read image from file
		img = cv2.imread(file_sample)
		#transform bgr image to grayscale
		#	grayscale can reduce the difficulty for training and calcluation
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#find face in this image
		faces = face_cascade.detectMultiScale(img, 1.3, 5)
		
		#if we detect only one face in this image, we consider this image is good
		#	and this face will be label as that person

		if(len(faces) == 1):

			#get the coordinate of face that in this image
			(x,y,w,h) = faces[0]

			#find the region of interest a.k.a the part of image that contains face
			roi = img[y:y+h, x:x+w]

			#Add data and label to training data set
			training_data.append(np.asarray(roi,dtype=np.uint8))

			#fisher face recognizer support int identifer, so we need a user_label and id label
			if name not in user_label:
				user_label.append(name)
				counter = counter + 1
			
			training_label.append(counter)

			
		else:
			print "Warning - File is not good:",file_sample

print "Start Training Data Set..."
recognizer.train(np.asarray(training_data),np.asarray(training_label))

print "Finish Training!"

print "Start Camera Caputure"
cap = cv2.VideoCapture(0)

print "...Start Detection and Recognization..."

font = cv2.FONT_HERSHEY_SIMPLEX
while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		#draw a rectangle for each detected face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

		#find the ROI of face
		roi = img[y:y+h,x:x+w]
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		# actual Recongnize function call
		[recognize_result,confidence] = recognizer.predict(gray)

		#find if we found match result
		if(recognize_result>=0 and abs(confidence - 100)<50 ):
			#print confidence
			cv2.putText(img,"%s-%3.2f%%" %(user_label[recognize_result -1],confidence),(x,y),font,.8,(255,255,255),1)
		else:
			print "someone else",recognize_result,confidence
		#"%s - %d -  %3.2f%%" %(user_label[recognize_result -1],recognize_result,confidence)else:
			#cv2.putText(img," %d -  %3.2f%%" %(recognize_result,confidence),(x,y),font,.8,(255,255,255),1)

	cv2.imshow('Face Recognization',img)
	#press 'q' to exit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


print "Done!! Are You satisfied with the result?"
cv2.destroyAllWindows()
cap.release()
