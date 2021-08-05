# AI- Face Detection using openCV


import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces
#img = cv2.imread('1111.jpg')

#In Video or live feed
webcam = cv2.VideoCapture(0)
"""
# Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)





"""
#print(face_coordinates)

#Iterate forever over frames
while True:
    #### Read the current frame
    successful_frame_read, frame = webcam.read()

    #### Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    #In Picture:
    #cv2.rectangle(img, (130,58), (130+204, 58+204), (0,255,0), 2)
    #for live feed or clip:
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
    
    cv2.imshow('Talal Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if q or Q is pressed
    if key==81 or key==113:
        break

#Release the VideoCapture object     
webcam.release()
cv2.destroyAllWindows()

print("Code Completed")