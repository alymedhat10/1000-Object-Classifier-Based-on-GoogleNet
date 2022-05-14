import cv2
import numpy as np


model = "bvlc_googlenet.caffemodel"
protxt = "bvlc_googlenet.prototxt.txt"

net = cv2.dnn.readNetFromCaffe(protxt,model)

text_file = open("classification_classes_ILSVRC2012.txt","r")
lines = text_file.readlines()

frame = cv2.imread("photos\dog.JPG")
frame = cv2.resize(frame, (400,300) )

model_frame = cv2.resize(frame , (224,224) )
blobfromImage = cv2.dnn.blobFromImage( model_frame , 1 , (224,224) )
net.setInput(blobfromImage)
classifications = net.forward()

min_value,max_value,min_loc,max_loc = cv2.minMaxLoc(classifications)

class_probability = max_value
class_number = max_loc

if class_probability > 0.2:

    label = lines[ class_number[0] ][0:-1]

    cv2.putText(frame,label,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

else:

    label = "unknown"

cv2.imshow("frame",frame)
key = cv2.waitKey(1000000) & 0xFF

if key == ord("q"):
    cv2.destroyAllWindows()

    
