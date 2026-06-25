# face-detection-project
# this is the python code which detects the face but it is not 100 percent pricise but it can find
import cv2

# 1. Load the pre-trained face cascade model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Initialize video capture from the primary webcam (index 0)
cap = cv2.VideoCapture(0)

print("Starting video stream... Press 'q' to exit.")

while True:
    # Capture frame-by-frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current live frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding box rectangles around every face found
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the live window feed
    cv2.imshow('Face Detection - Live Webcam', frame)

    # Check if the user pressed the 'q' key to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and release system hardware resources back to OS
cap.release()
cv2.destroyAllWindows()
