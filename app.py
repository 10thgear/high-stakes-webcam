import cv2
from ultralytics import YOLO

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame reading was successful
    if not ret:
        print("Failed to grab frame.")
        break

    #results = model.track(frame, classes=0, conf=0.8, imgsz=480)
    results = model.predict(frame, show_labels=False, show_conf=False)
    cv2.imshow("Live Camera", results[0].plot())

    # Press 'q' on the keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
