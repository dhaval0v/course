import cv2, os

# Set up the face detection and datasets path
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
datasets = 'Dataset'  
sub_data = 'Dhaval sir'     

# Create the dataset directory if it doesn't exist
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

# Image dimensions
(width, height) = (130, 100)   

# Initialize the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)  # Ensure correct camera index

# Check if the webcam is opened correctly
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

count = 1
while count < 31:
    print(f"Capturing image {count}")
    ret, im = webcam.read()
    if not ret:
        print("Error: Failed to capture image.")
        continue
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f'{path}/{count}.png', face_resize)
        count += 1
    
    cv2.imshow('OpenCV', im)
    if cv2.waitKey(10) == 27:  # Exit on 'Esc' key
        break

webcam.release()
cv2.destroyAllWindows()
