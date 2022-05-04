import cv2

# Create the video object
video = cv2.VideoCapture('/Users/xriskaraplis/Desktop/Car_Pedestrian_Tracking/Tesla Dashcam Capture.mp4')

# Pre trained car and pedestrian classifiers
car_classifier= '/Users/xriskaraplis/Desktop/Car_Pedestrian_Tracking/Car_Detector.xml'
pedestrian_classifier = '/Users/xriskaraplis/Desktop/Car_Pedestrian_Tracking/haarcascade_fullbody.xml'

# Create car classifier
car_tracker = cv2.CascadeClassifier(car_classifier)

# Create pedestrian classifier
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)

# While the video is playing, loop through frames
while True:

    # Read current frame
    (read_successful, frame) = video.read()

    # Safety catch
    if read_successful:
        # Convert frame to greyscale (Needed to distinguish haar features)
        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        break

    # Detect cars AND pedestrians
    cars = car_tracker.detectMultiScale(greyscale_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(greyscale_frame)

    # Draw rectangles around detected cars and pedestrians
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2) 

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2) 

    # Display frames with objects detected
    cv2.imshow('Self Driving Car', frame)

    # Don't close automatically (Waits for a key input)
    key = cv2.waitKey(1)

    # Press q or Q to quit application
    if key==81 or key==113:
        break

# Release video capture object
video.release()