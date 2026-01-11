import cv2
from ultralytics import YOLO




bodypose_model = YOLO("yolo11n-pose.pt")  # load an official YOLO model




# Create a VideoCapture object, 0 for default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Optional: operations on the frame can go here (e.g., converting to grayscale)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Live Camera Feed', frame)

    # Run inference on the source
    results = bodypose_model(frame)  # generator of Results objects
    #print(results)
    for r in results:
        #r.show()

        temp_frame = r.plot()


        #akeypoints = r.keypoints.tolist()
        #xy = keypoints.xy
        #print("Keypoints dim = ", xy.shape)  # (N, K, 2)
        #print(xy)  # x, y coordinates of keypoints for first detection
        '''
            0 - Nose
            1 - Left Eye
            2 - Right Eye
            3 - Left Ear
            4 - Right Ear
            5 - Left Shoulder
            6 - Right Shoulder
            7 - Left Elbow
            8 - Right Elbow
            9 - Left Wrist
            10 - Right Wrist
            11 - Left Hip
            12 - Right Hip
            13 - Left Knee
            14 - Right Knee
            15 - Left Ankle
            16 - Right Ankle
        '''

        #point_left_shoulder = (xy[0][5][0].item().int(), xy[0][5][0].item().int() ) # .tolist()
        #point_right_shoulder = (xy[0][6][0].item(), xy[0][6][0].item() ) #].tolist()


        # Green color in BGR
        color = (0, 255, 0)

        # Line thickness of 9 px
        thickness = 9

        #print("Left shoulder = ", point_left_shoulder)
        #print("Right shoulder = ", point_right_shoulder)

        #frame = cv2.line(frame, point_left_shoulder, point_right_shoulder, color, thickness)
    
    cv2.imshow('Live Camera Feed', temp_frame)







    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
